"""Summarize a frozen teacher comparison without exposing labels to teachers."""

import argparse
from collections import Counter
import json
from pathlib import Path

from teacher.bridge import atomic_json


def read_attempt(directory):
    trajectory, evaluation = directory / 'trajectory.json', directory / 'evaluation.json'
    if not trajectory.exists() or not evaluation.exists():
        return None
    trace = json.loads(trajectory.read_text())
    scored = json.loads(evaluation.read_text())
    usage = Counter()
    usage_records = 0
    events = directory / 'codex-events.jsonl'
    if events.exists():
        for line in events.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # The active process may still be appending a line.
            if event.get('type') == 'turn.completed' and isinstance(event.get('usage'), dict):
                usage_records += 1
                usage.update({key: value for key, value in event['usage'].items()
                              if isinstance(value, (int, float))})
    return {'status': trace['status'], 'scores': scored['scores'],
            'metrics': trace['metrics'], 'codex_usage': dict(usage),
            'codex_usage_records': usage_records}


def aggregate(records):
    n = len(records)
    correct = sum(bool(r['scores']['answer_correct']) for r in records)
    grounded = sum(bool(r['scores']['answer_correct_with_evidence']) for r in records)
    usage = Counter()
    for record in records:
        usage.update(record['codex_usage'])
    return {'evaluated': n, 'correct': correct, 'correct_with_evidence': grounded,
            'accuracy': correct / n if n else None,
            'grounded_accuracy': grounded / n if n else None,
            'statuses': dict(Counter(r['status'] for r in records)),
            'mean_tool_calls': sum(r['metrics'].get('tool_calls', 0) for r in records) / n if n else None,
            'student_input_tokens': sum(r['metrics'].get('input_tokens', 0) for r in records),
            'student_output_tokens': sum(r['metrics'].get('output_tokens', 0) for r in records),
            'episode_seconds': sum(r['metrics'].get('elapsed_seconds', 0) for r in records),
            'episodes_with_codex_usage': sum(r['codex_usage_records'] > 0 for r in records),
            'recorded_codex_usage': dict(usage)}


def report(output):
    manifest = json.loads((output / 'batch.json').read_text())
    selection = manifest['selection']
    if not selection or 'questions' not in selection or 'baseline' not in selection:
        raise ValueError('A frozen comparison selection with baseline metadata is required')
    baseline = Path(selection['baseline'])
    previous = json.loads((baseline / 'batch.json').read_text())
    if previous['dataset_sha256'] != manifest['dataset_sha256']:
        raise ValueError('Baseline and comparison datasets differ')
    comparability = {'student_settings_equal': previous['settings'] == manifest['settings'],
        'code_equal': {name: previous['code_sha256'].get(name) == manifest['code_sha256'].get(name)
            for name in ('pilot/agent.py', 'pilot/tools.py', 'pilot/contract.py',
                         'teacher/bridge.py', 'teacher/client.py')}}
    records = {}
    old_records = {}
    questions = selection['questions']
    for row in questions:
        eid = row['episode_id']
        record = read_attempt(output / 'attempts' / eid)
        if record is not None:
            records[eid] = record
        if row['matched_baseline']:
            old = read_attempt(baseline / 'attempts' / eid)
            if old is None:
                raise ValueError('A selected baseline comparison attempt is missing')
            old_records[eid] = old
    paired = sorted(records.keys() & old_records.keys())
    family_results = {}
    for family in sorted({q['family'] for q in questions}):
        ids = [q['episode_id'] for q in questions if q['family'] == family]
        shared = [eid for eid in ids if eid in paired]
        family_results[family] = {
            'planned': len(ids),
            'sol': aggregate([records[eid] for eid in ids if eid in records]),
            'matched_sol': aggregate([records[eid] for eid in shared]),
            'matched_previous': aggregate([old_records[eid] for eid in shared])}
    summary_path = output / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    result = {'requested_teacher_model': manifest.get('requested_teacher_model'),
        'requested_reasoning_effort': manifest.get('requested_reasoning_effort'),
        'baseline_model_identity': selection['baseline_model_identity'],
        'complete': len(records) == len(questions) and bool(summary and summary['complete']),
        'planned': len(questions), 'batch_summary': summary, 'comparability': comparability,
        'overall': aggregate(list(records.values())),
        'matched_sol': aggregate([records[eid] for eid in paired]),
        'matched_previous': aggregate([old_records[eid] for eid in paired]),
        'by_family': family_results,
        'by_negative': {str(flag).lower(): aggregate([records[q['episode_id']] for q in questions
            if q['negative'] == flag and q['episode_id'] in records]) for flag in (False, True)},
        'matched_episode_ids': paired,
        'outcomes': [{'episode_id': q['episode_id'], 'family': q['family'],
            'negative': q['negative'], 'sol': records.get(q['episode_id']),
            'previous': old_records.get(q['episode_id'])} for q in questions],
        'limitations': [
            'Training diagnostic: two questions per family, not held-out performance.',
            'Previous model was not pinned; Astra High is the user identification, not verified log metadata.',
            'Requested Sol model and effort are recorded CLI overrides, not a server-returned model fingerprint.',
            'Evidence coverage does not prove complete search or justified absence; review before SFT.',
            'Codex usage is only reported for episodes with completion usage events; missing records are not zero cost.',
            'Student transcript tokens are distinct from actual Codex usage; neither measures account allowance percentage.']}
    atomic_json(output / 'comparison.json', result)

    def rate(stats):
        return f"{stats['correct_with_evidence']}/{stats['evaluated']}" if stats['evaluated'] else 'pending'

    lines = ['# Sol 6.1 High training comparison', '',
        f"Requested teacher: `{result['requested_teacher_model']}`, reasoning `{result['requested_reasoning_effort']}`.",
        f"Evaluated {len(records)}/{len(questions)}; complete: {result['complete']}.", '',
        f"Correct with required evidence: **{rate(result['overall'])}**.",
        f"On the {len(paired)} matched evaluated questions: Sol **{rate(result['matched_sol'])}**, "
        f"previous teacher **{rate(result['matched_previous'])}**.", '',
        '| Family | Sol grounded / evaluated | Matched Sol | Matched previous |',
        '| --- | ---: | ---: | ---: |']
    for family, stats in family_results.items():
        lines.append(f"| {family} | {rate(stats['sol'])} | {rate(stats['matched_sol'])} | {rate(stats['matched_previous'])} |")
    lines += ['', '## Interpretation', ''] + ['- ' + item for item in result['limitations']]
    lines += ['', 'Exact metrics, statuses, token-record coverage, and per-question comparisons are in `comparison.json`.', '']
    (output / 'COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps({key: result[key] for key in ('complete', 'planned', 'overall', 'matched_sol', 'matched_previous')}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.output)


if __name__ == '__main__':
    main()
