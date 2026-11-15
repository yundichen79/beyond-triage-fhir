"""Start a verified live FHIR session for one training question and a fresh teacher."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from dataset.common import file_digest
from dataset.live import interrupt_cleanup, run_job
from pilot.agent import MODEL_CACHE, MODEL_ID, MODEL_REVISION, Settings, run_agent
from pilot.scoring import score_episode
from teacher.bridge import FileTeacher, atomic_json
from teacher.prepare import ROOT, opaque_id, public_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True, help='Opaque ID from the public index')
    parser.add_argument('--session', type=Path, required=True, help='New teacher-visible directory')
    parser.add_argument('--output', type=Path, required=True, help='New operator-only output directory')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/processed/cohort-200')
    parser.add_argument('--timeout', type=int, default=1800, help='Seconds allowed per teacher response')
    args = parser.parse_args()
    session, output = args.session.resolve(), args.output.resolve()
    if session == output or session in output.parents or output in session.parents:
        parser.error('Public session and private output must be separate directory trees')
    if session.exists() or output.exists():
        parser.error('Session and output must both be new paths; existing evidence is never replaced')
    if args.timeout <= 0:
        parser.error('Timeout must be positive')
    rows = [json.loads(line) for line in (args.dataset / 'questions.jsonl').read_text().splitlines()
            if line.strip()]
    selected = [r for r in rows if r['split'] == 'train' and opaque_id(r) == args.episode]
    if len(selected) != 1:
        parser.error('Episode is not a unique training question')
    row = selected[0]
    packet = public_episode(row)
    # Tokenizer only: no model weights, GPU allocation, download, or paid API.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        cache_dir=str(MODEL_CACHE), local_files_only=True, trust_remote_code=False)
    session.mkdir(parents=True)
    output.mkdir(parents=True)
    atomic_json(session / 'episode.json', packet)
    shutil.copyfile(ROOT / 'teacher/client.py', session / 'teacher_client.py')
    shutil.copyfile(ROOT / 'teacher_workspace/TEACHER.md', session / 'AGENTS.md')
    atomic_json(session / 'state.json', {'status': 'starting_backend'})
    settings = Settings()
    atomic_json(output / 'provenance.json', {
        'episode_id': args.episode, 'dataset_id': row['id'], 'settings': asdict(settings),
        'dataset_sha256': file_digest(args.dataset / 'questions.jsonl'),
        'code_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in
            [ROOT / name for name in ('pilot/agent.py', 'pilot/tools.py', 'pilot/contract.py',
             'teacher/prepare.py', 'teacher/bridge.py', 'teacher/client.py', 'teacher/serve.py')]},
        'teacher_context': 'Fresh Codex session; host access isolation must be configured externally',
        'student_transcript_tokenizer': {'model': MODEL_ID, 'revision': MODEL_REVISION},
        'codex_actual_usage': None,
    })

    def after_verify(client, selected_rows, snapshot):
        client.strict_search = True
        with (output / 'raw-turns.jsonl').open('x') as raw_log:
            model = FileTeacher(session, tokenizer, settings, raw_log, args.timeout)
            episode = run_agent(row['question'], row['patient_id'], model, client, settings, settings.seed)
        atomic_json(output / 'trajectory.json', dict(episode, model=model.metadata,
                    episode_id=args.episode, system_prompt=packet['system_prompt'], tools=packet['tools']))
        # Score only after termination. This file never enters the public session.
        atomic_json(output / 'evaluation.json', {
            'scores': score_episode(row, episode), 'approved_for_sft': False,
            'review_required': 'Check anchors, unsupported filters, and absence completeness.'})
        atomic_json(session / 'state.json', {
            'status': episode['status'], 'final': episode['final'],
            'metrics': episode['metrics'],
            'note': 'Episode ended. Scoring is private; stop this teacher context.'})

    live_args = SimpleNamespace(output=output / 'live', dataset=args.dataset,
        bundles=None, snapshot=None, batch_size=100, callback_status='running_teacher',
        callback_label='interactive teacher episode')
    try:
        with interrupt_cleanup():
            run_job(live_args, rows=selected, after_verify=after_verify)
    except Exception:
        atomic_json(session / 'state.json', {'status': 'backend_failed',
            'note': 'Operator must inspect private job logs. Do not infer missing records.'})
        raise


if __name__ == '__main__':
    main()
