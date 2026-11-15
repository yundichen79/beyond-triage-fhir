"""Public, standard-library-only client; has no access to backend code or labels."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def show(session):
    state = json.loads((session / 'state.json').read_text())
    if state['status'] == 'awaiting_response':
        return json.loads((session / 'turn.json').read_text())
    return state


def respond(session, text, wait=30):
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
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        result = show(session)
        if result.get('status') != 'processing' and result.get('turn_id') != turn['turn_id']:
            return result
        time.sleep(0.2)
    return {'status': 'submitted', 'turn_id': turn['turn_id'],
            'next': 'Run status to retrieve the result; do not submit again.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'respond'))
    parser.add_argument('--session', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--text', help='Raw assistant tool-call JSON; otherwise read stdin')
    args = parser.parse_args()
    try:
        result = show(args.session) if args.action == 'status' else respond(
            args.session, args.text if args.text is not None else sys.stdin.read())
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Client error: {exc}\n')


if __name__ == '__main__':
    main()
