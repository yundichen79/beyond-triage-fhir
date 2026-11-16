"""SFT conversion: failed actions removed, per-turn examples, prefix check, dev split."""

import json
import unittest

from sft.data import clean, dev_patient, examples


def call(name, **arguments):
    return {'type': 'function', 'function': {'name': name, 'arguments': arguments}}


def tool(name, value):
    return {'role': 'tool', 'name': name, 'content': json.dumps(value)}


MESSAGES = [
    {'role': 'user', 'content': 'question'},
    {'role': 'assistant', 'content': 'not json'},
    {'role': 'user', 'content': 'Output format error: use tool_call blocks'},
    {'role': 'assistant', 'content': '', 'tool_calls': [call('search_fhir', resource_type='Encounter'),
                                                        call('search_fhir', resource_type='Condition')]},
    tool('search_fhir', {'resources': [{'id': 'e1'}]}),
    tool('search_fhir', {'error': 'params.encounter must contain Encounter/logical-id references'}),
    {'role': 'assistant', 'content': '', 'tool_calls': [call('finish', answer={}, evidence=['X/unseen'])]},
    tool('finish', {'error': 'Citations must refer to resources actually returned by tools'}),
    {'role': 'assistant', 'content': '', 'tool_calls': [call('finish', answer={}, evidence=['Encounter/e1'])]},
]


class TemplateTokenizer:
    """A minimal chat template with the same generation-prompt convention as Qwen."""

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):
        parts = []
        for message in messages:
            body = message.get('content') or ''
            if message.get('tool_calls'):
                body += ''.join('<tool_call>' + json.dumps(c['function']) + '</tool_call>' for c in message['tool_calls'])
            parts.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
        return ''.join(parts) + ('<|im_start|>assistant\n' if add_generation_prompt else '')

    def encode(self, text, add_special_tokens=False):
        return list(text)


class SftDataTests(unittest.TestCase):
    def test_failed_actions_are_removed_and_real_results_kept_in_order(self):
        cleaned, removed = clean(MESSAGES)
        self.assertEqual(dict(removed), {'malformed_turns': 1, 'failed_calls': 2, 'empty_turns': 1})
        self.assertEqual([m['role'] for m in cleaned], ['user', 'assistant', 'tool', 'assistant'])
        self.assertEqual([c['function']['arguments']['resource_type'] for c in cleaned[1]['tool_calls']], ['Encounter'])
        self.assertEqual(json.loads(cleaned[2]['content']), {'resources': [{'id': 'e1'}]})
        self.assertEqual(cleaned[3]['tool_calls'][0]['function']['arguments']['evidence'], ['Encounter/e1'])
        # A clean trajectory is unchanged.
        again, removed = clean(cleaned)
        self.assertEqual((again, dict(removed)), (cleaned, {}))

    def test_one_example_per_turn_with_prompt_prefix_and_turn_end(self):
        record = {'episode_id': 'e', 'patient_id': 'Patient/p1', 'family': 'f', 'complexity': 'simple',
                  'messages': MESSAGES}
        rows, removed = examples(record, 'SYSTEM', [], TemplateTokenizer(), 10_000)
        self.assertEqual([r['turn'] for r in rows], [0, 1])
        for row in rows:
            self.assertTrue(row['prompt'].endswith('<|im_start|>assistant\n'))
            self.assertTrue(row['completion'].endswith('<|im_end|>'))
            self.assertNotIn('error', row['completion'])
        self.assertIn('"finish"', rows[1]['completion'])
        self.assertIn('"resources"', rows[1]['prompt'])
        self.assertNotIn('Output format error', rows[1]['prompt'])
        with self.assertRaisesRegex(ValueError, 'context budget'):
            examples(record, 'SYSTEM', [], TemplateTokenizer(), 10)

    def test_trajectory_must_end_with_finish(self):
        record = {'episode_id': 'e', 'patient_id': 'Patient/p1', 'family': 'f', 'complexity': 'simple',
                  'messages': MESSAGES[:6]}
        with self.assertRaisesRegex(ValueError, 'finish'):
            examples(record, 'SYSTEM', [], TemplateTokenizer(), 10_000)

    def test_dev_split_is_by_patient_and_stable(self):
        patients = [f'Patient/{i}' for i in range(2000)]
        share = sum(map(dev_patient, patients)) / len(patients)
        self.assertTrue(0.07 < share < 0.13)
        self.assertEqual([dev_patient(p) for p in patients[:50]], [dev_patient(p) for p in patients[:50]])


if __name__ == '__main__':
    unittest.main()
