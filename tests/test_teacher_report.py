"""Comparison denominators and partial usage coverage remain explicit."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from teacher.report import report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record(base, eid, correct):
    output = base / 'attempts' / eid
    write(output / 'trajectory.json', {'status': 'completed', 'metrics': {'tool_calls': 2}})
    write(output / 'evaluation.json', {'scores': {'answer_correct': correct,
          'answer_correct_with_evidence': correct}})


class TeacherReportTests(unittest.TestCase):
    def test_partial_report_pairs_only_evaluated_questions_and_marks_missing_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            old, new = base / 'previous', base / 'sol'
            shared = {'dataset_sha256': 'fixture', 'settings': {}, 'code_sha256': {}}
            write(old / 'batch.json', shared)
            questions = [{'episode_id': eid, 'family': 'family', 'negative': flag,
                          'matched_baseline': True} for eid, flag in [('a', False), ('b', True)]]
            write(new / 'batch.json', dict(shared, requested_teacher_model='gpt-6.1-sol',
                requested_reasoning_effort='high', selection={'questions': questions,
                'baseline': str(old), 'baseline_model_identity': 'Unpinned'}))
            record(old, 'a', False)
            record(old, 'b', True)
            record(new, 'a', True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = report(new)
            self.assertFalse(result['complete'])
            self.assertEqual(result['overall']['evaluated'], 1)
            self.assertEqual(result['matched_previous']['correct_with_evidence'], 0)
            self.assertEqual(result['matched_previous']['evaluated'], 1)
            self.assertEqual(result['by_negative']['true']['grounded_accuracy'], None)
            self.assertEqual(result['overall']['episodes_with_teacher_usage'], 0)
            # Completing the second question changes both matched denominators.
            record(new, 'b', False)
            write(new / 'summary.json', {'complete': True})
            with contextlib.redirect_stdout(io.StringIO()):
                result = report(new)
            self.assertTrue(result['complete'])
            self.assertEqual(result['matched_previous']['evaluated'], 2)
            self.assertEqual(result['matched_new']['grounded_accuracy'], 0.5)

    def test_claude_usage_is_normalized_and_baseline_can_be_another_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            selection_baseline, sol, claude = base / 'previous', base / 'sol', base / 'claude'
            shared = {'dataset_sha256': 'fixture', 'settings': {}, 'code_sha256': {}}
            questions = [{'episode_id': 'a', 'family': 'family', 'negative': False,
                          'matched_baseline': False}]
            selection = {'questions': questions, 'baseline': str(selection_baseline),
                         'baseline_model_identity': 'Unpinned'}
            write(selection_baseline / 'batch.json', shared)
            write(sol / 'batch.json', dict(shared, requested_teacher_model='gpt-6.1-sol'))
            write(claude / 'batch.json', dict(shared, teacher_cli='claude', selection=selection,
                  requested_teacher_model='claude-sonnet-5-5', requested_reasoning_effort='high'))
            record(sol, 'a', False)
            record(claude, 'a', True)
            (sol / 'attempts/a/codex-events.jsonl').write_text('\n'.join(json.dumps(e) for e in [
                {'type': 'item.completed', 'item': {'type': 'command_execution'}},
                {'type': 'turn.completed', 'usage': {'input_tokens': 1000, 'cached_input_tokens': 800,
                                                     'output_tokens': 50}}]))
            (claude / 'attempts/a/claude-events.jsonl').write_text('\n'.join(json.dumps(e) for e in [
                {'type': 'assistant', 'message': {'content': [{'type': 'tool_use'}, {'type': 'tool_use'}]}},
                {'type': 'result', 'num_turns': 3, 'usage': {'input_tokens': 10,
                 'cache_read_input_tokens': 300, 'cache_creation_input_tokens': 90, 'output_tokens': 40}}]))
            with contextlib.redirect_stdout(io.StringIO()):
                result = report(claude, sol)
            self.assertEqual(result['baseline_model_identity'], 'gpt-6.1-sol')
            self.assertEqual(result['matched_episode_ids'], ['a'])
            self.assertEqual(result['matched_new']['recorded_teacher_usage'], {
                'shell_commands': 2, 'input_tokens': 400, 'cached_input_tokens': 300,
                'output_tokens': 40, 'model_requests': 3})
            self.assertEqual(result['matched_previous']['recorded_teacher_usage'], {
                'shell_commands': 1, 'input_tokens': 1000, 'cached_input_tokens': 800, 'output_tokens': 50})
            self.assertIn('claude-sonnet-5-5', (claude / 'COMPARISON.md').read_text())


if __name__ == '__main__':
    unittest.main()
