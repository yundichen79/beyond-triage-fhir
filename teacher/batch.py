"""Plan or generate training trajectories, with one fresh teacher process per question.

The teacher is a subscription CLI (`codex exec` or Claude Code `claude -p`), never a
per-token API: API-key environment overrides are removed and the login mode is checked.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

from dataset.common import digest, file_digest
from dataset.live import interrupt_cleanup, run_job
from student.agent import MODEL_CACHE, MODEL_ID, MODEL_REVISION, Settings, run_agent
from student.scoring import score_episode
from teacher.bridge import FileTeacher, atomic_json
from teacher.prepare import ROOT, opaque_id, public_episode

PROMPT_HEADER = (
    'You are authorized to run teacher_client.py status and respond in this directory '
    'to solve one question using its tools. Start with `python3 teacher_client.py status`; '
    'do not read any file first. Continue until the recorded episode ends, then stop. '
    'Do not inspect files outside this directory or any other conversation. Do not start '
    'another agent process. The operator handles the next question in a separate fresh '
    'context.'
)
TEACHERS = ('codex', 'claude')
CODEX_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max', 'ultra')
CLAUDE_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
# The client may wait up to 300 s for the backend; give the shell tool more than that.
CLAUDE_BASH_TIMEOUT_MS = '360000'
# After the episode ends the CLI writes a closing message and then its usage
# summary; killing it sooner loses the usage record.
EXIT_GRACE_SECONDS = 60
# A 5-hour subscription limit is waited out inside the job; anything else stops it.
RESET_MARGIN_SECONDS = 120
MAX_LIMIT_RETRIES = 3
# A teacher that ends its own run with the episode open fails that question only;
# several in a row point to a systemic problem and stop the batch.
MAX_CONSECUTIVE_ABANDONED = 3


def usage_limit(attempt_dir):
    """The usage-limit rejection a Claude attempt ended on: (window, resets_at epoch
    seconds), or None. window is 'five_hour', or 'other' when any longer window
    (e.g. the 7-day limit) is exhausted."""
    path = Path(attempt_dir) / 'claude-events.jsonl'
    if not path.exists():
        return None
    found = None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        info = event.get('rate_limit_info') if event.get('type') == 'rate_limit_event' else None
        if not isinstance(info, dict) or info.get('status') != 'rejected':
            continue
        windows = info.get('unifiedWindows') or {}
        longer = any(isinstance(window, dict) and window.get('status') == 'rejected'
                     for name, window in windows.items() if name != 'five_hour')
        kind = 'five_hour' if info.get('rateLimitType') == 'five_hour' and not longer else 'other'
        found = (kind, info.get('resetsAt'))
    return found


def abandoned_episode(attempt_dir):
    """True when Claude ended its own run normally while the episode was still open: a
    final non-error result, no permission denials and no usage-limit rejection. Auth,
    permission and limit failures are not abandonment."""
    path = Path(attempt_dir) / 'claude-events.jsonl'
    if not path.exists():
        return False
    result = None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'result':
            result = event
    return (result is not None and result.get('is_error') is False
            and not result.get('permission_denials') and usage_limit(attempt_dir) is None)


def limit_wait(limit, elapsed, now, max_run_seconds, episode_timeout, retries):
    """(seconds to sleep before retrying, None) or (None, stop reason)."""
    if limit is None:
        return None, 'teacher_runtime_error'
    kind, resets_at = limit
    if kind != 'five_hour':
        return None, 'usage_limit_longer_window'
    if not isinstance(resets_at, (int, float)) or isinstance(resets_at, bool):
        return None, 'usage_limit_unknown_reset'
    if retries >= MAX_LIMIT_RETRIES:
        return None, 'usage_limit_repeated'
    wait = max(0.0, resets_at - now) + RESET_MARGIN_SECONDS
    if elapsed + wait + episode_timeout > max_run_seconds:
        return None, 'usage_limit_beyond_run_budget'
    return wait, None


def teacher_instructions():
    return (ROOT / 'teacher/TEACHER.md').read_text()


def teacher_prompt():
    # The instructions travel in the launch prompt, so the session holds no
    # AGENTS.md: that would be loaded and read a second time.
    return PROMPT_HEADER + ' Your full instructions follow.\n\n' + teacher_instructions()


def codex_environment():
    environment = dict(os.environ)
    for key in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'AZURE_OPENAI_API_KEY', 'OPENAI_BASE_URL'):
        environment.pop(key, None)
    return environment


def claude_environment():
    # Any of these would route Claude Code to per-token API or cloud billing.
    environment = dict(os.environ)
    for key in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL',
                'ANTHROPIC_PROFILE', 'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX',
                'CLAUDE_CODE_USE_FOUNDRY', 'CLAUDE_CODE_OAUTH_TOKEN'):
        environment.pop(key, None)
    environment.update(BASH_DEFAULT_TIMEOUT_MS=CLAUDE_BASH_TIMEOUT_MS,
                       BASH_MAX_TIMEOUT_MS=CLAUDE_BASH_TIMEOUT_MS)
    return environment


def teacher_environment(teacher):
    return claude_environment() if teacher == 'claude' else codex_environment()


def require_chatgpt_login(executable):
    result = subprocess.run([str(executable), 'login', 'status'], capture_output=True,
                            text=True, env=codex_environment(), timeout=30)
    message = (result.stdout + '\n' + result.stderr).lower()
    if result.returncode or 'logged in using chatgpt' not in message:
        # Do not echo login status: API-key modes may include a credential fragment.
        raise ValueError('Batch generation requires existing ChatGPT sign-in; API-key mode is not authorized')


def require_claude_login(executable):
    result = subprocess.run([str(executable), 'auth', 'status'], capture_output=True,
                            text=True, env=claude_environment(), timeout=30)
    try:
        status = json.loads(result.stdout)
    except ValueError:
        status = {}
    # Do not echo the status: it contains account details.
    if (result.returncode or status.get('loggedIn') is not True
            or status.get('authMethod') != 'claude.ai' or status.get('apiProvider') != 'firstParty'):
        raise ValueError('Batch generation requires an existing claude.ai subscription login; '
                         'API-key and cloud-provider modes are not authorized')


def require_login(teacher, executable):
    (require_claude_login if teacher == 'claude' else require_chatgpt_login)(executable)


def select_rows(rows, previous=(), limit=None):
    training = {opaque_id(row): row for row in rows if row['split'] == 'train'}
    done = set()
    for directory in previous:
        # The first interactive attempt and batch attempts both use provenance.json.
        paths = [directory / 'provenance.json']
        paths += sorted((directory / 'attempts').glob('*/provenance.json'))
        for path in paths:
            if not path.is_file() or not (path.parent / 'evaluation.json').is_file():
                continue
            meta = json.loads(path.read_text())
            eid = meta['episode_id']
            if eid not in training:
                raise ValueError('Previous attempt does not belong to this training dataset')
            if meta.get('row_sha256') != digest(training[eid]):
                # Old single-question outputs predate per-row hashes. Check the
                # complete source dataset digest at the caller before accepting them.
                if 'row_sha256' in meta:
                    raise ValueError('Previous attempt used a different question or label version')
            trajectory = path.parent / 'trajectory.json'
            if trajectory.is_file() and json.loads(trajectory.read_text()).get('status') == 'model_error':
                # Usage limits and runtime failures are interruptions, not completed
                # attempts. Preserve their evidence and retry in a new session.
                continue
            done.add(eid)
    ordered = sorted((r for eid, r in training.items() if eid not in done), key=opaque_id)
    return ordered[:limit] if limit is not None else ordered


def selected_manifest_rows(rows, manifest, dataset_hash):
    if manifest.get('dataset_sha256') != dataset_hash:
        raise ValueError('Selection dataset fingerprint differs')
    ids = manifest.get('episode_ids')
    if not isinstance(ids, list) or not ids or any(not isinstance(eid, str) for eid in ids):
        raise ValueError('Selection must contain a nonempty list of episode IDs')
    if len(ids) != len(set(ids)):
        raise ValueError('Selection contains duplicate episode IDs')
    training = {opaque_id(row): row for row in rows if row['split'] == 'train'}
    if any(eid not in training for eid in ids):
        raise ValueError('Selection contains an unknown or non-training episode')
    return [training[eid] for eid in ids]


def codex_command(executable, session, teacher_model=None, reasoning_effort=None):
    # Grant only the runtime installation and this question's public directory.
    # Do not mount ~/.codex broadly: it contains auth and prior conversations.
    executable, session = Path(executable).resolve(), Path(session).resolve()
    filesystem = {':root': 'deny', ':minimal': 'read', str(session): 'write',
                  str(executable.parent.parent): 'read'}
    table = '{' + ','.join(json.dumps(k) + '=' + json.dumps(v) for k, v in filesystem.items()) + '}'
    command = [str(executable), '--ask-for-approval', 'never', 'exec',
            '--ephemeral', '--ignore-user-config', '--skip-git-repo-check',
            '--cd', str(session), '--json',
            '-c', 'default_permissions="teacher_batch"',
            '-c', 'permissions.teacher_batch.filesystem=' + table,
            '-c', 'permissions.teacher_batch.network.enabled=false',
            '-c', 'web_search="disabled"']
    if teacher_model:
        command += ['--model', teacher_model]
    if reasoning_effort:
        command += ['-c', 'model_reasoning_effort=' + json.dumps(reasoning_effort)]
    return command + [teacher_prompt()]


def claude_command(executable, teacher_model, reasoning_effort):
    # Our instructions replace Claude Code's default system prompt, and Bash is the
    # only tool, restricted to the public client. --safe-mode drops the operator's
    # CLAUDE.md, plugins, hooks, skills and MCP servers; nothing is persisted.
    # This is permission-based isolation, not an OS sandbox like the Codex profile.
    if not teacher_model or not reasoning_effort:
        raise ValueError('Claude teacher runs require an explicit model and effort')
    return [str(Path(executable).resolve()), '-p', PROMPT_HEADER,
            '--model', teacher_model, '--effort', reasoning_effort,
            '--system-prompt', teacher_instructions(),
            '--tools', 'Bash',
            '--allowedTools', 'Bash(python3 teacher_client.py *)',
            '--permission-mode', 'dontAsk', '--permission-prompts', 'none',
            '--safe-mode', '--strict-mcp-config', '--disable-slash-commands',
            '--no-session-persistence', '--output-format', 'stream-json', '--verbose']


def teacher_command(teacher, executable, session, teacher_model=None, reasoning_effort=None):
    if teacher == 'claude':
        return claude_command(executable, teacher_model, reasoning_effort)
    return codex_command(executable, session, teacher_model, reasoning_effort)


class CodexDriver:
    """Run one teacher CLI process for one question (Codex or Claude Code)."""

    def __init__(self, command, output, timeout, cwd, teacher='codex', grace=EXIT_GRACE_SECONDS):
        self.command, self.output, self.timeout, self.grace = command, output, timeout, grace
        self.cwd, self.teacher = cwd, teacher
        self.process, self.started, self.handles = None, None, []

    def start(self):
        if self.process is not None:
            self.check()
            return
        self.handles = [(self.output / f'{self.teacher}-{name}').open('x')
                        for name in ('events.jsonl', 'stderr.log')]
        self.started = time.monotonic()
        self.process = subprocess.Popen(self.command, stdin=subprocess.DEVNULL,
            stdout=self.handles[0], stderr=self.handles[1], start_new_session=True,
            cwd=str(self.cwd), env=teacher_environment(self.teacher))

    def check(self):
        if self.process is None:
            raise ValueError('Teacher process did not start')
        code = self.process.poll()
        if code is not None:
            raise ValueError(f'Teacher process exited before finishing the episode (exit {code})')
        if time.monotonic() - self.started > self.timeout:
            raise ValueError('Teacher exceeded the per-question wall-time limit')

    def close(self):
        if self.process is not None:
            try:
                self.process.wait(timeout=self.grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=5)
        for handle in self.handles:
            handle.close()


def attempt(row, client, tokenizer, settings, session, output, executable, timeout,
            teacher_model=None, reasoning_effort=None, teacher='codex'):
    packet = public_episode(row)
    session.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    atomic_json(session / 'episode.json', packet)
    shutil.copyfile(ROOT / 'teacher/client.py', session / 'teacher_client.py')
    command = teacher_command(teacher, executable, session, teacher_model, reasoning_effort)
    atomic_json(output / 'provenance.json', {
        'episode_id': opaque_id(row), 'dataset_id': row['id'], 'row_sha256': digest(row),
        'settings': asdict(settings), 'command': command, 'fresh_context': True,
        'teacher_cli': teacher,
        'teacher_model_selection': 'Explicit CLI override' if teacher_model else 'Codex executable default; no model override',
        'requested_teacher_model': teacher_model, 'requested_reasoning_effort': reasoning_effort,
        'authentication': ('Existing claude.ai subscription login required; API-key and cloud-provider '
                           'environment overrides removed') if teacher == 'claude' else
                          'Existing ChatGPT login required; API-key environment overrides removed',
        'student_transcript_tokenizer': {'model': MODEL_ID, 'revision': MODEL_REVISION},
        'actual_teacher_usage_source': f'{teacher}-events.jsonl when provided by the CLI',
    })
    driver = CodexDriver(command, output, timeout, session, teacher)
    try:
        with (output / 'raw-turns.jsonl').open('x') as log:
            model = FileTeacher(session, tokenizer, settings, log, timeout,
                                on_request=driver.start, check_alive=driver.check)
            model.metadata['model'] = ('fresh-claude-code-teacher' if teacher == 'claude'
                                       else 'fresh-codex-exec-teacher')
            model.metadata['requested_teacher_model'] = teacher_model
            model.metadata['requested_reasoning_effort'] = reasoning_effort
            episode = run_agent(row['question'], row['patient_id'], model, client, settings, settings.seed)
        atomic_json(output / 'trajectory.json', dict(episode, model=model.metadata,
                    episode_id=opaque_id(row), system_prompt=packet['system_prompt'], tools=packet['tools']))
        scores = score_episode(row, episode)
        atomic_json(output / 'evaluation.json', {'scores': scores, 'approved_for_sft': False,
            'review_required': 'Check anchors, unsupported filters, and absence completeness.'})
        atomic_json(session / 'state.json', {'status': episode['status'], 'final': episode['final'],
            'metrics': episode['metrics'], 'note': 'Episode ended. Stop this context; scores are private.'})
        return {'episode_id': opaque_id(row), 'status': episode['status'], 'scores': scores}
    except BaseException:
        atomic_json(session / 'state.json', {'status': 'backend_failed', 'note': 'Stop this context.'})
        raise
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run'))
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/processed/cohort-200')
    parser.add_argument('--sessions', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume-from', type=Path, action='append', default=[])
    parser.add_argument('--selection', type=Path, help='Frozen training episode selection manifest')
    parser.add_argument('--teacher', choices=TEACHERS, default='codex',
                        help='Subscription CLI that runs the teacher')
    parser.add_argument('--teacher-model', help='Explicit model ID; pair with --reasoning-effort')
    parser.add_argument('--reasoning-effort', choices=CODEX_EFFORTS)
    parser.add_argument('--limit', type=int, help='Optional cap; default is every remaining training question')
    parser.add_argument('--episode-timeout', type=int, default=900)
    parser.add_argument('--max-run-seconds', type=int, default=12600,
                        help='Soft run budget including backend setup; stop between episodes')
    parser.add_argument('--codex', default=shutil.which('codex'))
    parser.add_argument('--claude', default=shutil.which('claude'))
    args = parser.parse_args()
    if bool(args.teacher_model) != bool(args.reasoning_effort):
        parser.error('--teacher-model and --reasoning-effort must be supplied together')
    if args.teacher == 'claude' and not args.teacher_model:
        parser.error('--teacher claude requires an explicit --teacher-model and --reasoning-effort')
    if args.teacher == 'claude' and args.reasoning_effort not in CLAUDE_EFFORTS:
        parser.error(f'Claude effort must be one of {", ".join(CLAUDE_EFFORTS)}')
    executable = args.claude if args.teacher == 'claude' else args.codex
    if args.selection and (args.resume_from or args.limit):
        parser.error('--selection cannot be combined with --resume-from or --limit')
    if args.limit is not None and args.limit <= 0 or args.episode_timeout <= 0 or args.max_run_seconds <= 0:
        parser.error('Limits and timeouts must be positive')
    rows = [json.loads(line) for line in (args.dataset / 'questions.jsonl').read_text().splitlines() if line.strip()]
    dataset_hash = file_digest(args.dataset / 'questions.jsonl')
    for previous in args.resume_from:
        manifest = previous / 'batch.json'
        if not manifest.exists():
            manifest = previous / 'provenance.json'
        if not manifest.exists() or json.loads(manifest.read_text()).get('dataset_sha256') != dataset_hash:
            parser.error('Resume source missing or dataset fingerprint differs')
    selection = json.loads(args.selection.read_text()) if args.selection else None
    selected = (selected_manifest_rows(rows, selection, dataset_hash) if selection is not None
                else select_rows(rows, args.resume_from, args.limit))
    print(json.dumps({'training_questions': sum(r['split'] == 'train' for r in rows),
        'selected_questions': len(selected), 'selected_patients': len({r['patient_id'] for r in selected}),
        'fresh_context_per_question': True, 'teacher_cli': args.teacher,
        'requested_teacher_model': args.teacher_model,
        'requested_reasoning_effort': args.reasoning_effort}, indent=2), flush=True)
    if args.action == 'plan' or not selected:
        return
    if not args.sessions or not args.output or not executable:
        parser.error(f'run requires --sessions, --output, and an installed {args.teacher} executable')
    sessions, output = args.sessions.resolve(), args.output.resolve()
    if sessions == output or sessions in output.parents or output in sessions.parents:
        parser.error('Public sessions and private outputs must be separate trees')
    if sessions.exists() or output.exists():
        parser.error('Use new output/session paths; use --resume-from to skip previously recorded attempts')
    require_login(args.teacher, executable)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        cache_dir=str(MODEL_CACHE), local_files_only=True, trust_remote_code=False)
    sessions.mkdir(parents=True)
    output.mkdir(parents=True, mode=0o700)
    settings = Settings()
    started = time.monotonic()
    # attempted counts questions with a finished attempt; usage-limit interruptions
    # are counted separately and retried in a fresh session.
    summary = {'planned': len(selected), 'attempted': 0, 'correct_with_evidence': 0,
               'interrupted_attempts': 0, 'usage_limit_waits': 0, 'abandoned_attempts': 0,
               'complete': False, 'stop_reason': None}
    atomic_json(output / 'batch.json', {'dataset_sha256': dataset_hash,
        'episode_ids': [opaque_id(r) for r in selected], 'resume_from': [str(p) for p in args.resume_from],
        'settings': asdict(settings), 'episode_timeout': args.episode_timeout,
        'teacher_cli': args.teacher,
        'requested_teacher_model': args.teacher_model, 'requested_reasoning_effort': args.reasoning_effort,
        'selection': selection,
        'max_run_seconds': args.max_run_seconds,
        'code_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in
            [ROOT / name for name in ('teacher/batch.py', 'teacher/bridge.py', 'teacher/client.py',
                                     'teacher/TEACHER.md',
                                     'student/agent.py', 'student/tools.py', 'student/contract.py')]}})

    def after_verify(client, selected_rows, snapshot):
        client.strict_search = True
        abandoned_in_a_row = 0
        with (output / 'results.jsonl').open('x') as results:
            for row in selected_rows:
                eid = opaque_id(row)
                retries = 0
                while not summary['stop_reason']:
                    if time.monotonic() - started + args.episode_timeout > args.max_run_seconds:
                        summary['stop_reason'] = 'run_time_budget'
                        break
                    # A retry gets its own session and output; the interrupted attempt is kept.
                    name = eid if retries == 0 else f'{eid}-retry{retries}'
                    print(f'Teacher question {summary["attempted"] + 1}/{len(selected_rows)}: {name}', flush=True)
                    result = attempt(row, client, tokenizer, settings, sessions / name,
                        output / 'attempts' / name, executable, args.episode_timeout,
                        args.teacher_model, args.reasoning_effort, args.teacher)
                    results.write(json.dumps(dict(result, attempt=name)) + '\n')
                    results.flush()
                    os.fsync(results.fileno())
                    if result['status'] != 'model_error':
                        abandoned_in_a_row = 0
                        summary['attempted'] += 1
                        summary['correct_with_evidence'] += int(result['scores']['answer_correct_with_evidence'])
                        atomic_json(output / 'summary.json', summary)
                        break
                    if args.teacher == 'claude' and abandoned_episode(output / 'attempts' / name):
                        # A failed attempt for this question only; a later resume retries it.
                        abandoned_in_a_row += 1
                        summary['attempted'] += 1
                        summary['abandoned_attempts'] += 1
                        if abandoned_in_a_row >= MAX_CONSECUTIVE_ABANDONED:
                            summary['stop_reason'] = 'repeated_abandoned_episodes'
                        else:
                            print(f'Teacher ended its run with {eid} unfinished; continuing', flush=True)
                        atomic_json(output / 'summary.json', summary)
                        break
                    summary['interrupted_attempts'] += 1
                    # Only a 5-hour subscription limit is waited out. Auth, permission and
                    # other runtime problems still stop the batch instead of repeating.
                    limit = usage_limit(output / 'attempts' / name) if args.teacher == 'claude' else None
                    wait, stop = limit_wait(limit, time.monotonic() - started, time.time(),
                                            args.max_run_seconds, args.episode_timeout, retries)
                    if stop:
                        summary['stop_reason'] = stop
                    else:
                        summary['usage_limit_waits'] += 1
                        print(f'Usage limit (five_hour): waiting {wait / 60:.0f} min for the reset, '
                              f'then retrying {eid} in a fresh session', flush=True)
                    atomic_json(output / 'summary.json', summary)
                    if stop:
                        break
                    time.sleep(wait)
                    retries += 1
                if summary['stop_reason']:
                    break
        summary['complete'] = summary['attempted'] == summary['planned']
        summary['stop_reason'] = summary['stop_reason'] or 'all_selected_attempted'

    live_args = SimpleNamespace(output=output / 'live', dataset=args.dataset, bundles=None,
        snapshot=None, batch_size=100, callback_status='running_teacher_batch', callback_label='teacher batch')
    try:
        with interrupt_cleanup():
            run_job(live_args, rows=selected, after_verify=after_verify)
    except BaseException:
        summary['stop_reason'] = 'interrupted_or_backend_error'
        raise
    finally:
        atomic_json(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
