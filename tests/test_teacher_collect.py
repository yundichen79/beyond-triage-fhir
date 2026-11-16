"""Corpus collection: compatible runs only, shortest grounded attempt selected, no labels."""

import json
from pathlib import Path
import tempfile
import unittest

from dataset.common import file_digest
from teacher.collect import collect
from teacher.prepare import opaque_id
from test_teacher_batch import row

INTERFACE = {'pilot/agent.py': 'a', 'pilot/tools.py': 't', 'pilot/contract.py': 'c'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.rows = [dict(r, family='cbc_panel') for r in (row('a'), row('b'), row('c'), row('held-out', 'test'))]
        dataset = self.base / 'dataset'
        dataset.mkdir()
        (dataset / 'questions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        self.dataset, self.hash = dataset, file_digest(dataset / 'questions.jsonl')

    def tearDown(self):
        self.directory.cleanup()

    def run_dir(self, name, attempts, interface=INTERFACE, dataset_hash=None):
        run = self.base / name
        write(run / 'batch.json', {'dataset_sha256': dataset_hash or self.hash, 'code_sha256': interface})
        for folder, source, status, grounded, tool_calls in attempts:
            attempt = run / 'attempts' / folder
            write(attempt / 'provenance.json', {'episode_id': opaque_id(source), 'teacher_cli': 'claude',
                                                'requested_teacher_model': 'claude-sonnet-5-5'})
            write(attempt / 'trajectory.json', {'status': status, 'final': None, 'messages': [],
                  'metrics': {'tool_calls': tool_calls, 'llm_calls': tool_calls + 1},
                  'system_prompt': 'shared', 'tools': []})
            write(attempt / 'evaluation.json', {'scores': {'answer_correct_with_evidence': grounded}})
        return run

    def test_selects_shortest_grounded_attempt_and_skips_unfinished_ones(self):
        a, b, c = self.rows[:3]
        first = self.run_dir('run1', [('a', a, 'completed', True, 5), ('b', b, 'model_error', False, 1),
                                      ('c', c, 'completed', False, 2)])
        second = self.run_dir('run2', [('a', a, 'completed', True, 3), ('b-retry1', b, 'completed', True, 4)])
        summary = collect([first, second], self.dataset, self.base / 'corpus')
        self.assertEqual(summary['training_questions'], 3)
        self.assertEqual(summary['questions_with_finished_attempt'], 3)
        self.assertEqual(summary['finished_attempts'], 4)
        self.assertEqual(summary['skipped'], {'interrupted_or_abandoned': 1})
        self.assertEqual(summary['selected_trajectories'], 2)
        records = [json.loads(line) for line in (self.base / 'corpus/trajectories.jsonl').read_text().splitlines()]
        selected = {r['dataset_id']: r for r in records if r['selected']}
        self.assertEqual(sorted(selected), ['a', 'b'])
        self.assertEqual(selected['a']['metrics']['tool_calls'], 3)
        self.assertFalse(any(r['approved_for_sft'] for r in records))
        text = (self.base / 'corpus/trajectories.jsonl').read_text()
        self.assertNotIn('ground_truth', text)
        corpus = json.loads((self.base / 'corpus/corpus.json').read_text())
        self.assertEqual(corpus['student_interface_sha256'], INTERFACE)
        self.assertEqual(corpus['system_prompt'], 'shared')
        self.assertEqual(corpus['summary'], summary)
        self.assertEqual(sorted(p.name for p in (self.base / 'corpus').iterdir()), ['corpus.json', 'trajectories.jsonl'])

    def test_refuses_mixed_interfaces_or_datasets_and_existing_output(self):
        a = self.rows[0]
        first = self.run_dir('run1', [('a', a, 'completed', True, 3)])
        other_interface = self.run_dir('run2', [('a', a, 'completed', True, 3)], dict(INTERFACE, **{'pilot/tools.py': 'v5'}))
        other_dataset = self.run_dir('run3', [('a', a, 'completed', True, 3)], dataset_hash='different')
        with self.assertRaisesRegex(ValueError, 'interface'):
            collect([first, other_interface], self.dataset, self.base / 'mixed')
        with self.assertRaisesRegex(ValueError, 'dataset'):
            collect([first, other_dataset], self.dataset, self.base / 'other')
        (self.base / 'existing').mkdir()
        with self.assertRaisesRegex(ValueError, 'existing'):
            collect([first], self.dataset, self.base / 'existing')


if __name__ == '__main__':
    unittest.main()
