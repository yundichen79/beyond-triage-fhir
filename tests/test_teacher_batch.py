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
from student.agent import Settings
from teacher.batch import (CodexDriver, MAX_LIMIT_RETRIES, abandoned_episode, RESET_MARGIN_SECONDS, attempt, limit_wait, usage_limit, claude_command, claude_environment, codex_command,
                           codex_environment, require_chatgpt_login, require_claude_login,
                           select_rows, selected_manifest_rows)
from teacher.client import respond, show
from teacher.prepare import opaque_id
from test_dataset import fixture
from test_student import CBC_QUESTION, fake_client
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
        # Instructions travel in the launch prompt; no startup file reads are requested.
        self.assertIn('# Teacher instructions', command[-1])
        self.assertNotIn('Read AGENTS.md', command[-1])

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

    def test_claude_command_uses_subscription_cli_with_only_the_public_client(self):
        command = claude_command('/runtime/bin/claude', 'claude-sonnet-5-5', 'high')
        self.assertEqual(command[1:3], ['-p', command[2]])
        self.assertEqual(command[command.index('--model') + 1], 'claude-sonnet-5-5')
        self.assertEqual(command[command.index('--effort') + 1], 'high')
        self.assertIn('# Teacher instructions', command[command.index('--system-prompt') + 1])
        self.assertEqual(command[command.index('--tools') + 1], 'Bash')
        self.assertEqual(command[command.index('--allowedTools') + 1], 'Bash(python3 teacher_client.py *)')
        self.assertEqual(command[command.index('--permission-mode') + 1], 'dontAsk')
        for flag in ('--safe-mode', '--strict-mcp-config', '--no-session-persistence'):
            self.assertIn(flag, command)
        with self.assertRaises(ValueError):
            claude_command('/runtime/bin/claude', None, None)

    def test_claude_environment_and_login_refuse_paid_api_modes(self):
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'sentinel', 'CLAUDE_CODE_USE_BEDROCK': '1'}):
            environment = claude_environment()
        self.assertNotIn('ANTHROPIC_API_KEY', environment)
        self.assertNotIn('CLAUDE_CODE_USE_BEDROCK', environment)
        self.assertEqual(environment['BASH_DEFAULT_TIMEOUT_MS'], '360000')
        with patch('teacher.batch.subprocess.run') as run:
            run.return_value.returncode = 0
            run.return_value.stderr = ''
            run.return_value.stdout = json.dumps({'loggedIn': True, 'authMethod': 'claude.ai',
                                                  'apiProvider': 'firstParty'})
            require_claude_login('/claude')
            for status in ({'loggedIn': True, 'authMethod': 'api_key', 'apiProvider': 'firstParty'},
                           {'loggedIn': True, 'authMethod': 'claude.ai', 'apiProvider': 'bedrock'},
                           {'loggedIn': False}):
                run.return_value.stdout = json.dumps(status)
                with self.subTest(status=status), self.assertRaisesRegex(ValueError, 'subscription'):
                    require_claude_login('/claude')

    def test_usage_limit_is_read_from_claude_rate_limit_events(self):
        def event(status, kind, windows):
            return {'type': 'rate_limit_event', 'rate_limit_info': {'status': status, 'resetsAt': 1794798600,
                    'rateLimitType': kind, 'unifiedWindows': windows}}
        five_hour = event('rejected', 'five_hour', {
            'five_hour': {'utilization': 1, 'status': 'rejected', 'resetsAt': 1794798600},
            'seven_day': {'utilization': 0.1, 'status': 'allowed', 'resetsAt': 1795302000}})
        weekly = event('rejected', 'five_hour', {
            'five_hour': {'status': 'rejected'}, 'seven_day': {'status': 'rejected'}})
        warning = event('allowed_warning', 'five_hour', {})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.assertIsNone(usage_limit(base))
            for events, expected in (([warning], None), ([warning, five_hour], ('five_hour', 1794798600)),
                                     ([weekly], ('other', 1794798600))):
                (base / 'claude-events.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\nnot json')
                with self.subTest(expected=expected):
                    self.assertEqual(usage_limit(base), expected)

    def test_abandoned_episode_is_distinguished_from_auth_permission_and_limit_failures(self):
        rejected = {'type': 'rate_limit_event', 'rate_limit_info': {'status': 'rejected', 'resetsAt': 1,
                    'rateLimitType': 'five_hour', 'unifiedWindows': {}}}
        cases = (
            ([{'type': 'result', 'is_error': False, 'permission_denials': []}], True),
            ([{'type': 'result', 'is_error': False, 'permission_denials': [{'tool_name': 'Bash'}]}], False),
            ([{'type': 'result', 'is_error': True, 'permission_denials': []}], False),
            ([rejected, {'type': 'result', 'is_error': True}], False),
            ([{'type': 'assistant', 'message': {'content': []}}], False))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.assertFalse(abandoned_episode(base))
            for events, expected in cases:
                (base / 'claude-events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
                with self.subTest(events=events):
                    self.assertEqual(abandoned_episode(base), expected)

    def test_only_a_five_hour_limit_within_budget_is_waited_out(self):
        now, reset = 1000.0, 1000.0 + 3600
        wait, stop = limit_wait(('five_hour', reset), 600, now, 50000, 900, 0)
        self.assertEqual((wait, stop), (3600 + RESET_MARGIN_SECONDS, None))
        self.assertEqual(limit_wait(('five_hour', now - 5), 0, now, 50000, 900, 0)[0], RESET_MARGIN_SECONDS)
        for limit, elapsed, retries, reason in (
                (None, 0, 0, 'teacher_runtime_error'),
                (('other', reset), 0, 0, 'usage_limit_longer_window'),
                (('five_hour', None), 0, 0, 'usage_limit_unknown_reset'),
                (('five_hour', reset), 0, MAX_LIMIT_RETRIES, 'usage_limit_repeated'),
                (('five_hour', reset), 46000, 0, 'usage_limit_beyond_run_budget')):
            with self.subTest(reason=reason):
                self.assertEqual(limit_wait(limit, elapsed, now, 50000, 900, retries), (None, reason))

    def test_driver_lets_the_teacher_exit_before_killing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name, script, grace, killed in (('slow-exit', 'sleep 1; echo summary', 10, False),
                                                ('hung', 'sleep 30', 0.5, True)):
                output = base / name
                output.mkdir()
                driver = CodexDriver(['bash', '-c', script], output, 60, base, 'claude', grace=grace)
                driver.start()
                started = time.monotonic()
                driver.close()
                with self.subTest(name=name):
                    self.assertEqual(driver.process.returncode < 0, killed)
                    self.assertLess(time.monotonic() - started, 9)
                    if not killed:
                        self.assertIn('summary', (output / 'claude-events.jsonl').read_text())

    def test_complete_attempt_records_tools_but_keeps_scores_private(self):
        self.run_complete_attempt('codex', 'gpt-6.1-sol')

    def test_complete_claude_attempt_records_teacher_cli(self):
        self.run_complete_attempt('claude', 'claude-sonnet-5-5')

    def run_complete_attempt(self, teacher, model_id):
        sample = row()
        failures = []
        drivers = []

        class FakeDriver:
            def __init__(self, command, output, timeout, cwd, teacher='codex'):
                self.session = Path(cwd)
                self.thread = None
                drivers.append((command, teacher))

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
                    base / 'public', base / 'private', f'/runtime/bin/{teacher}', 5,
                    model_id, 'high', teacher)
            self.assertEqual(drivers[0][1], teacher)
            self.assertIn(model_id, drivers[0][0])
            self.assertFalse(failures)
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(result['scores']['answer_correct_with_evidence'])
            public = json.loads((base / 'public/state.json').read_text())
            self.assertNotIn('scores', public)
            self.assertFalse((base / 'public/evaluation.json').exists())
            self.assertFalse((base / 'public/AGENTS.md').exists())
            evaluation = json.loads((base / 'private/evaluation.json').read_text())
            self.assertFalse(evaluation['approved_for_sft'])
            trace = json.loads((base / 'private/trajectory.json').read_text())
            self.assertEqual(trace['metrics']['http_requests'], 3)
            provenance = json.loads((base / 'private/provenance.json').read_text())
            self.assertEqual(provenance['requested_teacher_model'], model_id)
            self.assertEqual(provenance['requested_reasoning_effort'], 'high')
            self.assertEqual(provenance['teacher_cli'], teacher)
            self.assertEqual(trace['model']['requested_teacher_model'], model_id)


if __name__ == '__main__':
    unittest.main()
