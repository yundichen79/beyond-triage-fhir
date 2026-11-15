"""Adapt a fresh interactive teacher to the unchanged student agent loop."""

import json
import os
from pathlib import Path
import time
import uuid

from pilot.agent import ContextLimit, Generation


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


class FileTeacher:
    def __init__(self, session, tokenizer, settings, raw_log, timeout=1800,
                 on_request=None, check_alive=None):
        self.session = Path(session)
        self.tokenizer = tokenizer
        self.settings = settings
        self.raw_log = raw_log
        self.timeout = timeout
        self.on_request = on_request
        self.check_alive = check_alive
        # Messages already delivered to the teacher; each turn exposes only newer ones.
        self.delivered = 0
        self.turns = 0
        self.metadata = {
            'model': 'interactive-codex-teacher',
            'actual_model_identity': 'Record in operator run notes; not observable by the bridge',
            'token_accounting': 'Qwen student transcript tokens; Codex usage is unavailable',
            'outer_chat_context_enforced': False,
        }

    def seed(self, value):
        # The bridge cannot set Codex's sampling seed. Never claim that it did.
        self.requested_student_seed = value

    def generate(self, messages, tools, *, max_new_tokens=None):
        budget = min(self.settings.max_new_tokens, max_new_tokens or self.settings.max_new_tokens)
        rendered = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        count = len(self.tokenizer.encode(rendered, add_special_tokens=False))
        if count + budget > self.settings.max_context_tokens:
            raise ContextLimit('Student-rendered transcript exceeds the configured context budget')
        turn_id = uuid.uuid4().hex
        # Preserve actual tool responses even if the job is interrupted while
        # waiting for the next teacher action; this private journal is append-only.
        self.raw_log.write(json.dumps({'type': 'request', 'turn_id': turn_id,
            'messages': messages, 'tools': tools, 'student_input_tokens': count,
            'max_response_tokens': budget}) + '\n')
        self.raw_log.flush()
        os.fsync(self.raw_log.fileno())
        self.turns += 1
        # The teacher's own assistant messages are not echoed back to it.
        new_messages = [m for m in messages[self.delivered:] if m.get('role') != 'assistant']
        self.delivered = len(messages)
        atomic_json(self.session / 'turn.json', {
            'status': 'awaiting_response', 'turn_id': turn_id,
            'turn': self.turns, 'max_turns': self.settings.max_steps,
            'new_messages': new_messages,
            'messages': messages, 'tools': tools,
            'student_input_tokens': count, 'max_response_tokens': budget,
            'student_context_tokens': count,
            'student_context_available': self.settings.max_context_tokens - budget,
            'token_accounting': self.metadata['token_accounting'],
        })
        atomic_json(self.session / 'state.json', {'status': 'awaiting_response', 'turn_id': turn_id})
        if self.on_request:
            self.on_request()
        deadline = time.monotonic() + self.timeout
        response_path = self.session / 'response.json'
        while time.monotonic() < deadline:
            if response_path.exists():
                try:
                    if response_path.stat().st_size > 2 * 1024 * 1024:
                        raise ValueError('Teacher response exceeds transport limit')
                    response = json.loads(response_path.read_text())
                except json.JSONDecodeError:
                    time.sleep(0.1)
                    continue
                response_path.unlink()
                if (set(response) != {'turn_id', 'text'} or response['turn_id'] != turn_id
                        or not isinstance(response['text'], str)):
                    raise ValueError('Stale or malformed teacher transport response')
                atomic_json(self.session / 'state.json', {'status': 'processing', 'turn_id': turn_id})
                ids = self.tokenizer.encode(response['text'], add_special_tokens=False)
                truncated = len(ids) > budget
                accepted = self.tokenizer.decode(ids[:budget]) if truncated else response['text']
                self.raw_log.write(json.dumps({'type': 'response', 'turn_id': turn_id, 'raw_text': response['text'],
                    'accepted_text': accepted, 'student_input_tokens': count,
                    'student_output_tokens': min(len(ids), budget), 'truncated': truncated}) + '\n')
                self.raw_log.flush()
                os.fsync(self.raw_log.fileno())
                return Generation(accepted, count, min(len(ids), budget), truncated)
            if self.check_alive:
                self.check_alive()
            time.sleep(0.2)
        raise ValueError('Teacher response timeout')
