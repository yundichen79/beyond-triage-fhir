"""Public, standard-library-only client; has no access to backend code or labels.

`status` and `respond` print a compact view: the turn ID, budget counters and only
the tool results added since the previous turn. The first turn also carries the
shared rules and tool definitions, once. `--full` prints the complete turn.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Backend states in which the next turn is still being prepared.
PENDING = ('processing', 'starting_backend')
SEEN = 'client_seen.json'
DEFAULT_WAIT = 300


def show(session):
    """Return the raw pending turn, or the session state when no turn is pending."""
    try:
        state = json.loads((session / 'state.json').read_text())
    except FileNotFoundError:
        return {'status': 'starting_backend'}
    if state['status'] == 'awaiting_response':
        return json.loads((session / 'turn.json').read_text())
    return state


def wait_for_turn(session, previous=None, wait=DEFAULT_WAIT):
    """Block until a turn other than `previous` is pending or the episode ends."""
    deadline = time.monotonic() + wait
    while True:
        result = show(session)
        if result.get('status') not in PENDING and (
                previous is None or result.get('turn_id') != previous):
            return result
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.2)


def action_text(payload):
    """A JSON array of calls becomes consecutive <tool_call> blocks, the student format
    for several calls in one turn; anything else is submitted unchanged."""
    try:
        value = json.loads(payload)
    except ValueError:
        return payload
    if isinstance(value, list) and value and all(isinstance(call, dict) for call in value):
        return '\n'.join('<tool_call>' + json.dumps(call, ensure_ascii=False) + '</tool_call>'
                         for call in value)
    return payload


def respond(session, text, wait=DEFAULT_WAIT):
    state = json.loads((session / 'state.json').read_text())
    if state['status'] != 'awaiting_response':
        raise ValueError('Session is not awaiting a response; run status')
    turn = json.loads((session / 'turn.json').read_text())
    if len(text.encode()) > 1024 * 1024:
        raise ValueError('Response exceeds 1 MiB')
    response = json.dumps({'turn_id': turn['turn_id'], 'text': text}).encode()
    # An exclusive file prevents duplicate submissions; the server only consumes
    # a complete, parseable object for the currently pending turn.
    fd = os.open(session / 'response.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(response)
        handle.flush()
        os.fsync(handle.fileno())
    result = wait_for_turn(session, turn['turn_id'], wait)
    if result is not None:
        return result
    return {'status': 'submitted', 'turn_id': turn['turn_id'],
            'next': 'Run status to wait for the result; do not submit again.'}


def compact_tools(tools):
    """Tool definitions without the answer schema, which the question already shows."""
    compact = []
    for tool in tools:
        function = dict(tool['function'])
        if function['name'] == 'finish':
            parameters = json.loads(json.dumps(function['parameters']))
            if 'answer' in parameters.get('properties', {}):
                parameters['properties']['answer'] = {
                    'type': 'object', 'description': 'Must match the Answer JSON Schema in the question'}
            function['parameters'] = parameters
        compact.append(function)
    return compact


def compact(turn):
    """Delta view of a pending turn; other states pass through unchanged."""
    if turn.get('status') != 'awaiting_response' or 'new_messages' not in turn:
        return turn
    view = {key: turn[key] for key in ('turn_id', 'status', 'turn', 'max_turns',
            'student_context_tokens', 'student_context_available') if key in turn}
    new = []
    for message in turn['new_messages']:
        if message['role'] == 'system':
            view['rules'] = message['content']
            view['tools'] = compact_tools(turn['tools'])
        elif message['role'] == 'tool':
            try:
                result = json.loads(message['content'])
            except (TypeError, ValueError):
                result = message['content']
            new.append({'tool': message.get('name'), 'result': result})
        elif message['role'] == 'user':
            new.append({'user': message['content']})
    view['new'] = new
    return view


def already_seen(session, turn_id):
    try:
        return json.loads((session / SEEN).read_text()).get('turn_id') == turn_id
    except (OSError, ValueError):
        return False


def mark_seen(session, turn_id):
    (session / SEEN).write_text(json.dumps({'turn_id': turn_id}) + '\n')


def render(session, result, full=False, repeat=False):
    if result.get('status') != 'awaiting_response':
        return result
    turn_id = result['turn_id']
    if not (full or repeat) and already_seen(session, turn_id):
        return {'turn_id': turn_id, 'status': 'awaiting_response',
                'note': 'This turn was already shown. Submit your response to it with respond; '
                        'use status --repeat only if its contents were lost.'}
    mark_seen(session, turn_id)
    return result if full else compact(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'respond'))
    parser.add_argument('--session', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('payload', nargs='?',
                        help="respond: one call as JSON, or a JSON array of calls, in single quotes")
    parser.add_argument('--text', help='Raw assistant tool-call JSON; otherwise read stdin')
    parser.add_argument('--full', action='store_true', help='Print the complete turn, including history')
    parser.add_argument('--repeat', action='store_true', help='Show a turn again in compact form')
    parser.add_argument('--wait', type=int, default=DEFAULT_WAIT,
                        help='Seconds to wait for the backend before returning')
    args = parser.parse_args()
    try:
        if args.action == 'status':
            result = wait_for_turn(args.session, wait=args.wait) or show(args.session)
        else:
            if args.payload is not None and args.text is not None:
                raise ValueError('Give the action once, as the argument or with --text')
            text = args.payload if args.payload is not None else args.text
            result = respond(args.session, action_text(text if text is not None else sys.stdin.read()),
                             wait=args.wait)
        print(json.dumps(render(args.session, result, args.full, args.repeat),
                         ensure_ascii=False, separators=(',', ':')))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Client error: {exc}\n')


if __name__ == '__main__':
    main()
