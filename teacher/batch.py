"""Plan or generate training trajectories, with one fresh Codex process per question."""

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
from pilot.agent import MODEL_CACHE, MODEL_ID, MODEL_REVISION, Settings, run_agent
from pilot.scoring import score_episode
from teacher.bridge import FileTeacher, atomic_json
from teacher.prepare import ROOT, opaque_id, public_episode

PROMPT = (
    'Read AGENTS.md and episode.json in this directory. You are authorized to run '
    'teacher_client.py status and respond to solve this one question using its tools. '
    'Continue until the recorded episode ends, then stop. Do not inspect files outside '
    'this directory or any other conversation. Do not start another Codex process. '
    'The operator handles the next question in a separate fresh context.'
)


def codex_environment():
    environment = dict(os.environ)
    for key in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'AZURE_OPENAI_API_KEY', 'OPENAI_BASE_URL'):
        environment.pop(key, None)
    return environment


def require_chatgpt_login(executable):
    result = subprocess.run([str(executable), 'login', 'status'], capture_output=True,
                            text=True, env=codex_environment(), timeout=30)
    message = (result.stdout + '\n' + result.stderr).lower()
    if result.returncode or 'logged in using chatgpt' not in message:
        # Do not echo login status: API-key modes may include a credential fragment.
        raise ValueError('Batch generation requires existing ChatGPT sign-in; API-key mode is not authorized')


def select_rows(rows, previous=(), limit=None):
    training = {opaque_id(row): row for row in rows if row['split'] == 'train'}
    done = set()
    for directory in previous:
        # Both the original one-question pilot and batch attempts use provenance.json.
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
    return command + [PROMPT]


class CodexDriver:
    def __init__(self, command, output, timeout):
        self.command, self.output, self.timeout = command, output, timeout
        self.process, self.started, self.handles = None, None, []

    def start(self):
        if self.process is not None:
            self.check()
            return
        self.handles = [(self.output / name).open('x') for name in ('codex-events.jsonl', 'codex-stderr.log')]
        self.started = time.monotonic()
        self.process = subprocess.Popen(self.command, stdin=subprocess.DEVNULL,
            stdout=self.handles[0], stderr=self.handles[1], start_new_session=True,
            cwd=self.command[self.command.index('--cd') + 1], env=codex_environment())

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
                self.process.wait(timeout=3)
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
            teacher_model=None, reasoning_effort=None):
    packet = public_episode(row)
    session.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    atomic_json(session / 'episode.json', packet)
    shutil.copyfile(ROOT / 'teacher/client.py', session / 'teacher_client.py')
    shutil.copyfile(ROOT / 'teacher_workspace/TEACHER.md', session / 'AGENTS.md')
    command = codex_command(executable, session, teacher_model, reasoning_effort)
    atomic_json(output / 'provenance.json', {
        'episode_id': opaque_id(row), 'dataset_id': row['id'], 'row_sha256': digest(row),
        'settings': asdict(settings), 'command': command, 'fresh_context': True,
        'teacher_model_selection': 'Explicit CLI override' if teacher_model else 'Codex executable default; no model override',
        'requested_teacher_model': teacher_model, 'requested_reasoning_effort': reasoning_effort,
        'authentication': 'Existing ChatGPT login required; API-key environment overrides removed',
        'student_transcript_tokenizer': {'model': MODEL_ID, 'revision': MODEL_REVISION},
        'actual_codex_usage_source': 'codex-events.jsonl when provided by the CLI',
    })
    driver = CodexDriver(command, output, timeout)
    try:
        with (output / 'raw-turns.jsonl').open('x') as log:
            model = FileTeacher(session, tokenizer, settings, log, timeout,
                                on_request=driver.start, check_alive=driver.check)
            model.metadata['model'] = 'fresh-codex-exec-teacher'
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
    parser.add_argument('--teacher-model', help='Explicit Codex model ID; pair with --reasoning-effort')
    parser.add_argument('--reasoning-effort', choices=('low', 'medium', 'high', 'xhigh', 'max', 'ultra'))
    parser.add_argument('--limit', type=int, help='Optional cap; default is every remaining training question')
    parser.add_argument('--episode-timeout', type=int, default=900)
    parser.add_argument('--max-run-seconds', type=int, default=12600,
                        help='Soft run budget including backend setup; stop between episodes')
    parser.add_argument('--codex', default=shutil.which('codex'))
    args = parser.parse_args()
    if bool(args.teacher_model) != bool(args.reasoning_effort):
        parser.error('--teacher-model and --reasoning-effort must be supplied together')
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
        'fresh_context_per_question': True, 'requested_teacher_model': args.teacher_model,
        'requested_reasoning_effort': args.reasoning_effort}, indent=2), flush=True)
    if args.action == 'plan' or not selected:
        return
    if not args.sessions or not args.output or not args.codex:
        parser.error('run requires --sessions, --output, and an installed codex executable')
    sessions, output = args.sessions.resolve(), args.output.resolve()
    if sessions == output or sessions in output.parents or output in sessions.parents:
        parser.error('Public sessions and private outputs must be separate trees')
    if sessions.exists() or output.exists():
        parser.error('Use new output/session paths; use --resume-from to skip previously recorded attempts')
    require_chatgpt_login(args.codex)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        cache_dir=str(MODEL_CACHE), local_files_only=True, trust_remote_code=False)
    sessions.mkdir(parents=True)
    output.mkdir(parents=True, mode=0o700)
    settings = Settings()
    started = time.monotonic()
    summary = {'planned': len(selected), 'attempted': 0, 'correct_with_evidence': 0,
               'complete': False, 'stop_reason': None}
    atomic_json(output / 'batch.json', {'dataset_sha256': dataset_hash,
        'episode_ids': [opaque_id(r) for r in selected], 'resume_from': [str(p) for p in args.resume_from],
        'settings': asdict(settings), 'episode_timeout': args.episode_timeout,
        'requested_teacher_model': args.teacher_model, 'requested_reasoning_effort': args.reasoning_effort,
        'selection': selection,
        'max_run_seconds': args.max_run_seconds,
        'code_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in
            [ROOT / name for name in ('teacher/batch.py', 'teacher/bridge.py', 'teacher/client.py',
                                     'pilot/agent.py', 'pilot/tools.py', 'pilot/contract.py')]}})

    def after_verify(client, selected_rows, snapshot):
        client.strict_search = True
        with (output / 'results.jsonl').open('x') as results:
            for row in selected_rows:
                if time.monotonic() - started + args.episode_timeout > args.max_run_seconds:
                    summary['stop_reason'] = 'run_time_budget'
                    break
                eid = opaque_id(row)
                print(f'Teacher question {summary["attempted"] + 1}/{len(selected_rows)}: {eid}', flush=True)
                result = attempt(row, client, tokenizer, settings, sessions / eid,
                    output / 'attempts' / eid, args.codex, args.episode_timeout,
                    args.teacher_model, args.reasoning_effort)
                results.write(json.dumps(result) + '\n')
                results.flush()
                os.fsync(results.fileno())
                summary['attempted'] += 1
                summary['correct_with_evidence'] += int(result['scores']['answer_correct_with_evidence'])
                atomic_json(output / 'summary.json', summary)
                # Auth/permission/runtime problems must not trigger hundreds of
                # identical paid/account-consuming retries on subsequent questions.
                if result['status'] == 'model_error':
                    summary['stop_reason'] = 'teacher_runtime_error'
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
