"""Batch selection, isolation arguments, and complete bridge execution without a model call."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from dataset.common import Snapshot, digest
from dataset.oracle import evaluate
from pilot.agent import Settings
from teacher.batch import (attempt, codex_command, codex_environment, require_chatgpt_login,
                           select_rows, selected_manifest_rows)
from teacher.client import respond, show
from teacher.prepare import opaque_id
from test_dataset import fixture
from test_pilot import CBC_QUESTION, fake_client
from test_teacher import TinyTokenizer


def row(name='a', split='train'):
    answer, evidence = evaluate(Snapshot(fixture()), 'Patient/p1',
                               {'operation': 'cbc_panel', 'as_of': '2026-10-02'})
    return dict(id=name, split=split, question=CBC_QUESTION, patient_id='Patient/p1',
                ground_truth_answer=answer, ground_truth_resource_ids=evidence)


class BatchTests(unittest.TestCase):
    def test_frozen_selection_rejects_changed_dataset_heldout_and_duplicate_ids(self):
        rows = [row('a'), row('b'), row('c', 'validation'), row('d', 'test')]
        manifest = {'dataset_sha256': 'fixture', 'episode_ids': [opaque_id(rows[1]), opaque_id(rows[0])]}
        self.assertEqual([r['id'] for r in selected_manifest_rows(rows, manifest, 'fixture')], ['b', 'a'])
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            selected_manifest_rows(rows, manifest, 'changed')
        for ids in ([], [opaque_id(rows[2])], [opaque_id(rows[3])], ['unknown'],
                    [opaque_id(rows[0]), opaque_id(rows[0])]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                selected_manifest_rows(rows, dict(manifest, episode_ids=ids), 'fixture')

    def test_explicit_teacher_model_and_effort_reach_codex(self):
        command = codex_command('/runtime/bin/codex', '/tmp/episode', 'gpt-6.1-sol', 'high')
        self.assertEqual(command[command.index('--model') + 1], 'gpt-6.1-sol')
        self.assertIn('model_reasoning_effort="high"', command)
        self.assertIn('--ignore-user-config', command)

    def test_selection_excludes_heldout_and_completed_prior_attempts(self):
        rows = [row('a'), row('b'), row('c', 'validation'), row('d', 'test')]
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory)
            (previous / 'provenance.json').write_text(json.dumps({
                'episode_id': opaque_id(rows[0]), 'row_sha256': digest(rows[0])}))
            self.assertEqual(len(select_rows(rows, [previous])), 2)
            (previous / 'evaluation.json').write_text('{}')
            self.assertEqual([r['id'] for r in select_rows(rows, [previous])], ['b'])
            changed = dict(rows[0], question=rows[0]['question'] + ' CHANGED')
            with self.assertRaises(ValueError):
                select_rows([changed, *rows[1:]], [previous])

    def test_command_starts_fresh_and_only_grants_session_and_runtime(self):
        command = codex_command('/home/example/runtime/bin/codex', '/tmp/teacher-question')
        self.assertIn('--ephemeral', command)
        self.assertIn('--ignore-user-config', command)
        self.assertNotIn('resume', command)
        self.assertNotIn('fork', command)
        self.assertNotIn('--dangerously-bypass-approvals-and-sandbox', command)
        policy = next(v for v in command if v.startswith('permissions.teacher_batch.filesystem='))
        self.assertIn('"/tmp/teacher-question"="write"', policy)
        self.assertIn('"/home/example/runtime"="read"', policy)
        self.assertIn('":root"="deny"', policy)
        self.assertNotIn('ground_truth', ' '.join(command))

    def test_resume_retries_runtime_errors_but_preserves_scored_attempts(self):
        rows = [row('a'), row('b'), row('c')]
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory)
            for sample, status in zip(rows, ('model_error', 'completed', 'step_limit')):
                output = previous / 'attempts' / opaque_id(sample)
                output.mkdir(parents=True)
                (output / 'provenance.json').write_text(json.dumps({
                    'episode_id': opaque_id(sample), 'row_sha256': digest(sample)}))
                (output / 'evaluation.json').write_text('{"approved_for_sft": false}')
                (output / 'trajectory.json').write_text(json.dumps({'status': status}))
            self.assertEqual([r['id'] for r in select_rows(rows, [previous])], ['a'])
            # A later successful retry still counts, whichever order sources use.
            retry = previous / 'retry'
            retry.mkdir()
            (retry / 'provenance.json').write_text(json.dumps({
                'episode_id': opaque_id(rows[0]), 'row_sha256': digest(rows[0])}))
            (retry / 'evaluation.json').write_text('{}')
            (retry / 'trajectory.json').write_text('{"status": "completed"}')
            self.assertEqual(select_rows(rows, [previous, retry]), [])
            self.assertEqual(select_rows(rows, [retry, previous]), [])

    def test_api_key_environment_removed_and_non_chatgpt_login_rejected(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sentinel', 'CODEX_API_KEY': 'sentinel'}):
            self.assertNotIn('OPENAI_API_KEY', codex_environment())
            self.assertNotIn('CODEX_API_KEY', codex_environment())
        with patch('teacher.batch.subprocess.run') as run:
            run.return_value.returncode = 0
            run.return_value.stdout = 'Logged in using an API key - sentinel'
            run.return_value.stderr = ''
            with self.assertRaisesRegex(ValueError, 'requires existing ChatGPT'):
                require_chatgpt_login('/codex')

    def test_complete_attempt_records_tools_but_keeps_scores_private(self):
        sample = row()
        failures = []

        class FakeDriver:
            def __init__(self, command, output, timeout):
                self.session = Path(command[command.index('--cd') + 1])
                self.thread = None

            def start(self):
                if self.thread:
                    return
                def act():
                    try:
                        calls = [
                            {'name': 'search_fhir', 'arguments': {'resource_type': 'DiagnosticReport',
                                'params': {'_id': 'cbc1'}}},
                            {'name': 'read_fhir', 'arguments': {'reference': 'Observation/hb1'}},
                            {'name': 'read_fhir', 'arguments': {'reference': 'Observation/plt1'}},
                            {'name': 'finish', 'arguments': {'answer': sample['ground_truth_answer'],
                                'evidence': sample['ground_truth_resource_ids']}},
                        ]
                        previous = None
                        for call in calls:
                            deadline = time.monotonic() + 5
                            while time.monotonic() < deadline:
                                current = show(self.session)
                                if current.get('status') == 'awaiting_response' and current['turn_id'] != previous:
                                    break
                                time.sleep(0.01)
                            else:
                                raise AssertionError('Missing fresh turn')
                            previous = current['turn_id']
                            respond(self.session, json.dumps(call), wait=0)
                    except BaseException as exc:
                        failures.append(exc)
                self.thread = threading.Thread(target=act)
                self.thread.start()

            def check(self):
                if failures:
                    raise ValueError(str(failures[0]))

            def close(self):
                if self.thread:
                    self.thread.join(5)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch('teacher.batch.CodexDriver', FakeDriver):
                result = attempt(sample, fake_client(), TinyTokenizer(), Settings(),
                    base / 'public', base / 'private', '/runtime/bin/codex', 5,
                    'gpt-6.1-sol', 'high')
            self.assertFalse(failures)
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(result['scores']['answer_correct_with_evidence'])
            public = json.loads((base / 'public/state.json').read_text())
            self.assertNotIn('scores', public)
            self.assertFalse((base / 'public/evaluation.json').exists())
            evaluation = json.loads((base / 'private/evaluation.json').read_text())
            self.assertFalse(evaluation['approved_for_sft'])
            trace = json.loads((base / 'private/trajectory.json').read_text())
            self.assertEqual(trace['metrics']['http_requests'], 3)
            provenance = json.loads((base / 'private/provenance.json').read_text())
            self.assertEqual(provenance['requested_teacher_model'], 'gpt-6.1-sol')
            self.assertEqual(provenance['requested_reasoning_effort'], 'high')
            self.assertEqual(trace['model']['requested_teacher_model'], 'gpt-6.1-sol')


if __name__ == '__main__':
    unittest.main()
