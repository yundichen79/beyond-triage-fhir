"""Collect finished teacher attempts from batch outputs into one trajectory corpus.

Only runs with identical student-interface code and the same dataset are combined.
Interrupted and abandoned attempts (status model_error) are skipped. For each training
question, the shortest grounded attempt (fewest tool calls, then fewest turns) is marked
`selected`. Hidden labels are not copied, only scores. `selected` is not approval for SFT:
every trajectory still needs review of anchors, filters and absence claims.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from dataset.common import file_digest
from dataset.questions import negative
from teacher.bridge import atomic_json
from teacher.prepare import ROOT, opaque_id
from teacher.report import teacher_usage

# Files that define what the student sees; runs that differ here are not combined.
INTERFACE_FILES = ('pilot/agent.py', 'pilot/tools.py', 'pilot/contract.py')


def load_run(run):
    manifest = json.loads((run / 'batch.json').read_text())
    for attempt in sorted((run / 'attempts').glob('*')):
        paths = [attempt / name for name in ('provenance.json', 'trajectory.json', 'evaluation.json')]
        if all(path.is_file() for path in paths):
            provenance, trajectory, evaluation = (json.loads(path.read_text()) for path in paths)
            yield manifest, attempt, provenance, trajectory, evaluation


def collect(runs, dataset, output):
    if output.exists():
        raise ValueError(f'Refusing to replace an existing corpus: {output}')
    dataset_hash = file_digest(dataset / 'questions.jsonl')
    rows = {opaque_id(row): row for row in map(json.loads, (dataset / 'questions.jsonl').read_text().splitlines())
            if row['split'] == 'train'}
    interface, shared, records, skipped = None, None, [], Counter()
    for run in runs:
        manifest = json.loads((run / 'batch.json').read_text())
        if manifest['dataset_sha256'] != dataset_hash:
            raise ValueError(f'{run} used a different dataset')
        hashes = {name: manifest['code_sha256'].get(name) for name in INTERFACE_FILES}
        if interface is None:
            interface = hashes
        elif hashes != interface:
            raise ValueError(f'{run} used a different student interface; do not mix interface versions')
        for _, attempt, provenance, trajectory, evaluation in load_run(run):
            if trajectory['status'] == 'model_error':
                skipped['interrupted_or_abandoned'] += 1
                continue
            row = rows.get(provenance['episode_id'])
            if row is None:
                raise ValueError(f'{attempt} is not a training question in this dataset')
            prompt = {'system_prompt': trajectory['system_prompt'], 'tools': trajectory['tools']}
            if shared is None:
                shared = prompt
            elif prompt != shared:
                raise ValueError(f'{attempt} saw a different system prompt or tool set')
            scores = evaluation['scores']
            usage, _ = teacher_usage(attempt)
            records.append({
                'episode_id': provenance['episode_id'], 'dataset_id': row['id'], 'family': row['family'],
                'complexity': row.get('complexity'), 'negative_case': negative(row['ground_truth_answer'], row['family']),
                'question': row['question'], 'patient_id': row['patient_id'],
                'answer_contract': trajectory.get('answer_contract'), 'status': trajectory['status'],
                'grounded': bool(scores['answer_correct_with_evidence']), 'scores': scores,
                'final': trajectory['final'], 'metrics': trajectory['metrics'],
                'messages': trajectory['messages'], 'teacher': {
                    'cli': provenance.get('teacher_cli', 'codex'),
                    'model': provenance.get('requested_teacher_model'),
                    'effort': provenance.get('requested_reasoning_effort'), 'usage': usage},
                'source': str(attempt.relative_to(ROOT) if attempt.is_relative_to(ROOT) else attempt),
                'approved_for_sft': False, 'selected': False})
    by_question = defaultdict(list)
    for record in records:
        by_question[record['episode_id']].append(record)
    for attempts in by_question.values():
        grounded = [r for r in attempts if r['grounded']]
        if grounded:
            min(grounded, key=lambda r: (r['metrics']['tool_calls'], r['metrics']['llm_calls']))['selected'] = True
    families = defaultdict(Counter)
    for row in rows.values():
        families[row['family']]['training_questions'] += 1
    for eid, attempts in by_question.items():
        family = families[attempts[0]['family']]
        family['covered'] += 1
        family['selected'] += any(r['selected'] for r in attempts)
        for record in attempts:
            family['attempts'] += 1
            family[record['status']] += 1
    selected = [r for r in records if r['selected']]
    summary = {
        'training_questions': len(rows), 'questions_with_finished_attempt': len(by_question),
        'finished_attempts': len(records), 'grounded_attempts': sum(r['grounded'] for r in records),
        'selected_trajectories': len(selected), 'skipped': dict(skipped),
        'statuses': dict(Counter(r['status'] for r in records)),
        'selected_by_family': {f: dict(c) for f, c in sorted(families.items())},
        'selected_mean_tool_calls': sum(r['metrics']['tool_calls'] for r in selected) / len(selected) if selected else None,
        'negative_cases_selected': sum(r['negative_case'] for r in selected)}
    output.mkdir(parents=True, mode=0o700)
    with (output / 'trajectories.jsonl').open('x') as handle:
        for record in sorted(records, key=lambda r: (r['episode_id'], r['source'])):
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    atomic_json(output / 'manifest.json', {
        'dataset_sha256': dataset_hash, 'runs': [str(run) for run in runs],
        'student_interface_sha256': interface, 'system_prompt': shared['system_prompt'] if shared else None,
        'tools': shared['tools'] if shared else None,
        'selection_policy': 'Per question: shortest grounded attempt (fewest tool calls, then fewest turns).',
        'review_required': 'Check anchors, unsupported filters and absence claims before SFT.'})
    atomic_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True, help='Teacher batch output folders')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/processed/cohort-200')
    parser.add_argument('--output', type=Path, required=True, help='New corpus folder')
    args = parser.parse_args()
    print(json.dumps(collect(args.runs, args.dataset, args.output), indent=2))


if __name__ == '__main__':
    main()
