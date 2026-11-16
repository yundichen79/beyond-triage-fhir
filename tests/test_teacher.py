"""Public export isolation and genuine teacher-to-student-loop integration."""

import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import subprocess
import sys

from pilot.agent import SYSTEM_PROMPT, Settings, parse_calls, run_agent
from teacher.bridge import FileTeacher, atomic_json
from teacher.client import action_text, render, respond, show, wait_for_turn
from teacher.prepare import public_episode
from test_pilot import CBC_QUESTION, fake_client


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return 'rendered transcript'

    def encode(self, value, **kwargs):
        return list(value.encode())

    def decode(self, value):
        return bytes(value).decode(errors='replace')


def await_turn(session):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if (session / 'state.json').exists():
            result = show(session)
            if result.get('status') == 'awaiting_response':
                return result
        time.sleep(0.01)
    raise AssertionError('Teacher turn did not arrive')


class TeacherTests(unittest.TestCase):
    def test_export_uses_allowlist_and_refuses_heldout(self):
        row = dict(id='cbc_panel_PRIVATE', split='train', question=CBC_QUESTION,
                   patient_id='Patient/p1', ground_truth_answer={'SECRET': 123},
                   ground_truth_resource_ids=['Observation/SECRET'], oracle_spec='SECRET',
                   family='SECRET', complexity='SECRET')
        packet = public_episode(row)
        self.assertNotIn('SECRET', json.dumps(packet))
        self.assertNotIn('PRIVATE', json.dumps(packet))
        # v6: the schema is exported once; the finish tool is the same for every question.
        self.assertIn('properties', packet['answer_schema'])
        self.assertNotIn('properties', packet['tools'][-1]['function']['parameters']['properties']['answer'])
        for split in ('test', 'validation'):
            with self.assertRaises(ValueError):
                public_episode(dict(row, split=split))

    def test_live_loop_adapter_executes_and_records_real_tool_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            settings = Settings(max_steps=2)
            with (session / 'raw.jsonl').open('w') as log:
                model = FileTeacher(session, TinyTokenizer(), settings, log, timeout=5)
                result = {}
                def run():
                    result.update(run_agent(CBC_QUESTION, 'Patient/p1', model,
                                            fake_client(), settings, 1))
                thread = threading.Thread(target=run)
                thread.start()
                first = await_turn(session)
                respond(session, json.dumps({'name': 'calculate', 'arguments': {
                    'earlier': '1', 'later': '2.25', 'decimal_places': 1}}), wait=0)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    second = show(session)
                    if second.get('turn_id') not in (None, first['turn_id']):
                        break
                    time.sleep(0.01)
                else:
                    self.fail('Second turn did not arrive')
                response = json.loads(second['messages'][-1]['content'])
                self.assertEqual(response['value'], '1.2')
                respond(session, json.dumps({'name': 'read_fhir', 'arguments': {
                    'reference': 'Observation/not-seen'}}), wait=0)
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(result['status'], 'step_limit')
                self.assertEqual(result['metrics']['tool_calls'], 2)
                self.assertEqual(result['metrics']['http_requests'], 0)
                self.assertEqual(result['metrics']['tool_errors'], 1)
                journal = [json.loads(line) for line in (session / 'raw.jsonl').read_text().splitlines()]
                self.assertEqual([r['type'] for r in journal], ['request', 'response', 'request', 'response'])
                self.assertEqual(json.loads(journal[2]['messages'][-1]['content'])['value'], '1.2')

    def test_compact_turns_show_static_context_once_and_only_new_results(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            settings = Settings(max_steps=2)
            with (session / 'raw.jsonl').open('w') as log:
                model = FileTeacher(session, TinyTokenizer(), settings, log, timeout=5)
                thread = threading.Thread(target=run_agent, args=(
                    CBC_QUESTION, 'Patient/p1', model, fake_client(), settings, 1))
                thread.start()
                first = render(session, wait_for_turn(session, wait=5))
                self.assertEqual(first['turn'], 1)
                self.assertEqual(first['rules'], SYSTEM_PROMPT)
                self.assertNotIn('messages', first)
                finish = next(t for t in first['tools'] if t['name'] == 'finish')
                self.assertNotIn('properties', finish['parameters']['properties']['answer'])
                self.assertEqual(len(first['new']), 1)
                self.assertIn('Answer JSON Schema', first['new'][0]['user'])
                # A second status for the same turn does not reprint it.
                again = render(session, show(session))
                self.assertEqual(set(again), {'turn_id', 'status', 'note'})
                self.assertIn('rules', render(session, show(session), repeat=True))
                self.assertEqual(len(render(session, show(session), full=True)['messages']), 2)
                # respond blocks and returns the next turn directly.
                second = render(session, respond(session, json.dumps({'name': 'calculate', 'arguments': {
                    'earlier': '1', 'later': '2.25', 'decimal_places': 1}}), wait=5))
                self.assertEqual(second['turn'], 2)
                self.assertNotIn('rules', second)
                self.assertNotIn('tools', second)
                self.assertEqual(len(second['new']), 1)
                self.assertEqual(second['new'][0]['tool'], 'calculate')
                self.assertEqual(second['new'][0]['result']['value'], '1.2')
                respond(session, json.dumps({'name': 'read_fhir', 'arguments': {
                    'reference': 'Observation/not-seen'}}), wait=0)
                thread.join(5)
                self.assertFalse(thread.is_alive())
                # The launcher writes the terminal state; the client passes it through.
                final = {'status': 'step_limit', 'final': None, 'note': 'Episode ended.'}
                atomic_json(session / 'state.json', final)
                self.assertEqual(render(session, wait_for_turn(session, wait=1)), final)

    def test_single_line_shell_argument_and_call_arrays_reach_the_student_parser(self):
        self.assertEqual(action_text('{"name":"x","arguments":{}}'), '{"name":"x","arguments":{}}')
        self.assertEqual(action_text('not json'), 'not json')
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            atomic_json(session / 'state.json', {'status': 'awaiting_response', 'turn_id': 't1'})
            atomic_json(session / 'turn.json', {'status': 'awaiting_response', 'turn_id': 't1'})
            client = Path(__file__).resolve().parents[1] / 'teacher/client.py'
            command = (f"{sys.executable} {client} --session {session} --wait 0 respond "
                       """'[{"name":"read_fhir","arguments":{"reference":"Observation/a"}},"""
                       """{"name":"search_fhir","arguments":{"resource_type":"Condition","""
                       """"params":{"code:text":"Alzheimer\\u0027s"}}}]'""")
            done = subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=30)
            self.assertEqual(done.returncode, 0, done.stderr)
            submitted = json.loads((session / 'response.json').read_text())
            calls = parse_calls(submitted['text'])
            self.assertEqual([c['name'] for c in calls], ['read_fhir', 'search_fhir'])
            self.assertEqual(calls[1]['arguments']['params']['code:text'], "Alzheimer's")

    def test_context_budget_blocks_before_teacher_input(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            settings = Settings(max_context_tokens=21, max_new_tokens=4)
            model = FileTeacher(session, TinyTokenizer(), settings, io.StringIO(), timeout=1)
            result = run_agent(CBC_QUESTION, 'Patient/p1', model, fake_client(), settings, 1)
            self.assertEqual(result['status'], 'context_limit')
            self.assertFalse((session / 'turn.json').exists())

    def test_stale_response_fails_without_executing_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            atomic_json(session / 'response.json', {'turn_id': 'stale', 'text': '{}'})
            with (session / 'raw.jsonl').open('w') as log:
                model = FileTeacher(session, TinyTokenizer(), Settings(), log, timeout=1)
                result = run_agent(CBC_QUESTION, 'Patient/p1', model, fake_client(), Settings(), 1)
            self.assertEqual(result['status'], 'model_error')
            self.assertEqual(result['metrics']['tool_calls'], 0)


if __name__ == '__main__':
    unittest.main()
