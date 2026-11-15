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
            self.assertEqual(result['overall']['episodes_with_codex_usage'], 0)
            # Completing the second question changes both matched denominators.
            record(new, 'b', False)
            write(new / 'summary.json', {'complete': True})
            with contextlib.redirect_stdout(io.StringIO()):
                result = report(new)
            self.assertTrue(result['complete'])
            self.assertEqual(result['matched_previous']['evaluated'], 2)
            self.assertEqual(result['matched_sol']['grounded_accuracy'], 0.5)


if __name__ == '__main__':
    unittest.main()
