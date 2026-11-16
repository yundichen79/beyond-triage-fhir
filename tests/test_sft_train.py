"""SFT training data logic: tokenization boundary, limits and schedule (no GPU needed)."""

import unittest

from sft.train import Config, schedule, tokenize


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class MergingTokenizer(CharTokenizer):
    """Merges 'ab' into one token, so a prompt ending in 'a' and a completion starting
    with 'b' tokenize differently apart than joined."""

    def encode(self, text, add_special_tokens=False):
        return [1000 if piece == 'ab' else ord(piece) for piece in self._pieces(text)]

    @staticmethod
    def _pieces(text):
        pieces, index = [], 0
        while index < len(text):
            if text[index:index + 2] == 'ab':
                pieces.append('ab'); index += 2
            else:
                pieces.append(text[index]); index += 1
        return pieces


class SftTrainTests(unittest.TestCase):
    def test_completion_start_marks_the_first_trained_token(self):
        rows = tokenize([{'id': 'x', 'prompt': 'PROMPT', 'completion': '<tool_call>{}<|im_end|>'}], CharTokenizer(), 100)
        self.assertEqual(rows[0]['start'], len('PROMPT'))
        self.assertEqual(len(rows[0]['input_ids']), len('PROMPT<tool_call>{}<|im_end|>'))

    def test_boundary_length_and_empty_completion_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'boundary'):
            tokenize([{'id': 'x', 'prompt': 'xa', 'completion': 'by'}], MergingTokenizer(), 100)
        with self.assertRaisesRegex(ValueError, 'exceed'):
            tokenize([{'id': 'x', 'prompt': 'p' * 10, 'completion': 'c'}], CharTokenizer(), 5)
        with self.assertRaisesRegex(ValueError, 'empty'):
            tokenize([{'id': 'x', 'prompt': 'p', 'completion': ''}], CharTokenizer(), 5)

    def test_schedule_counts_optimizer_steps(self):
        self.assertEqual(schedule(1098, Config()), (69, 138, 4))
        self.assertEqual(schedule(10, Config(epochs=3, effective_batch=4, warmup_ratio=0.0)), (3, 9, 1))


if __name__ == '__main__':
    unittest.main()
