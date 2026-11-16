"""Summarize a frozen teacher comparison without exposing labels to teachers."""

import argparse
from collections import Counter
import json
from pathlib import Path

from teacher.bridge import atomic_json


def events(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue  # The active process may still be appending a line.


def teacher_usage(directory):
    """Normalized teacher CLI usage: total input (cached included), cached input,
    output and shell commands, from Codex or Claude Code event streams."""
    usage, records = Counter(), 0
    for event in events(directory / 'codex-events.jsonl'):
        if event.get('type') == 'turn.completed' and isinstance(event.get('usage'), dict):
            records += 1
            raw = event['usage']  # Codex input_tokens already include cached tokens.
            usage.update(input_tokens=raw.get('input_tokens', 0),
                         cached_input_tokens=raw.get('cached_input_tokens', 0),
                         output_tokens=raw.get('output_tokens', 0))
        item = event.get('item') or {}
        if event.get('type') == 'item.completed' and item.get('type') == 'command_execution':
            usage['shell_commands'] += 1
    requests, has_result = {}, False
    for event in events(directory / 'claude-events.jsonl'):
        if event.get('type') == 'assistant':
            # One API request can span several assistant events with the same message id;
            # num_turns counts tool turns, not requests. Keep each request's latest usage.
            message = event.get('message') or {}
            requests[message.get('id')] = message.get('usage') or {}
            content = (event.get('message') or {}).get('content') or []
            usage['shell_commands'] += sum(block.get('type') == 'tool_use' for block in content
                                           if isinstance(block, dict))
        if event.get('type') == 'result' and isinstance(event.get('usage'), dict):
            records += 1
            has_result = True
            raw = event['usage']  # Claude input_tokens exclude cache reads and writes.
            cached = raw.get('cache_read_input_tokens', 0) or 0
            usage.update(input_tokens=(raw.get('input_tokens', 0) or 0) + cached
                         + (raw.get('cache_creation_input_tokens', 0) or 0),
                         cached_input_tokens=cached, output_tokens=raw.get('output_tokens', 0) or 0)
    if requests:
        usage['model_requests'] = len(requests)
    if requests and not has_result:
        # The process ended before its usage summary. Per-request input and cache
        # counts are complete; per-request output counts are not, so output is kept
        # separately as a lower bound and the fallback is counted.
        records += 1
        cached = sum(u.get('cache_read_input_tokens') or 0 for u in requests.values())
        usage.update(input_tokens=sum((u.get('input_tokens') or 0) + (u.get('cache_read_input_tokens') or 0)
                                      + (u.get('cache_creation_input_tokens') or 0) for u in requests.values()),
                     cached_input_tokens=cached,
                     output_tokens_lower_bound=sum(u.get('output_tokens') or 0 for u in requests.values()),
                     usage_from_requests=1)
    return dict(usage), records


def read_attempt(directory):
    trajectory, evaluation = directory / 'trajectory.json', directory / 'evaluation.json'
    if not trajectory.exists() or not evaluation.exists():
        return None
    trace = json.loads(trajectory.read_text())
    scored = json.loads(evaluation.read_text())
    usage, records = teacher_usage(directory)
    return {'status': trace['status'], 'scores': scored['scores'],
            'metrics': trace['metrics'], 'teacher_usage': usage,
            'teacher_usage_records': records}


def aggregate(records):
    n = len(records)
    correct = sum(bool(r['scores']['answer_correct']) for r in records)
    grounded = sum(bool(r['scores']['answer_correct_with_evidence']) for r in records)
    usage = Counter()
    with_usage = [r for r in records if r['teacher_usage_records'] > 0]
    for record in with_usage:
        usage.update(record['teacher_usage'])
    return {'evaluated': n, 'correct': correct, 'correct_with_evidence': grounded,
            'accuracy': correct / n if n else None,
            'grounded_accuracy': grounded / n if n else None,
            'statuses': dict(Counter(r['status'] for r in records)),
            'mean_tool_calls': sum(r['metrics'].get('tool_calls', 0) for r in records) / n if n else None,
            'student_input_tokens': sum(r['metrics'].get('input_tokens', 0) for r in records),
            'student_output_tokens': sum(r['metrics'].get('output_tokens', 0) for r in records),
            'episode_seconds': sum(r['metrics'].get('elapsed_seconds', 0) for r in records),
            'episodes_with_teacher_usage': len(with_usage),
            'recorded_teacher_usage': dict(usage),
            'mean_teacher_usage_per_episode': {key: value / len(with_usage) for key, value in usage.items()}
                if with_usage else None}


def report(output, baseline=None):
    manifest = json.loads((output / 'batch.json').read_text())
    selection = manifest['selection']
    if not selection or 'questions' not in selection or 'baseline' not in selection:
        raise ValueError('A frozen comparison selection with baseline metadata is required')
    # The selection's baseline marks which questions it attempted; another run on the
    # same selection (e.g. a different teacher) is compared wherever both evaluated.
    override = baseline is not None
    baseline = Path(baseline) if override else Path(selection['baseline'])
    previous = json.loads((baseline / 'batch.json').read_text())
    if previous['dataset_sha256'] != manifest['dataset_sha256']:
        raise ValueError('Baseline and comparison datasets differ')
    comparability = {'student_settings_equal': previous['settings'] == manifest['settings'],
        'code_equal': {name: previous['code_sha256'].get(name) == manifest['code_sha256'].get(name)
            for name in ('pilot/agent.py', 'pilot/tools.py', 'pilot/contract.py',
                         'teacher/bridge.py', 'teacher/client.py', 'teacher_workspace/TEACHER.md')}}
    records = {}
    old_records = {}
    questions = selection['questions']
    for row in questions:
        eid = row['episode_id']
        record = read_attempt(output / 'attempts' / eid)
        if record is not None:
            records[eid] = record
        if override or row['matched_baseline']:
            old = read_attempt(baseline / 'attempts' / eid)
            if old is None and not override:
                raise ValueError('A selected baseline comparison attempt is missing')
            if old is not None:
                old_records[eid] = old
    paired = sorted(records.keys() & old_records.keys())
    family_results = {}
    for family in sorted({q['family'] for q in questions}):
        ids = [q['episode_id'] for q in questions if q['family'] == family]
        shared = [eid for eid in ids if eid in paired]
        family_results[family] = {
            'planned': len(ids),
            'new': aggregate([records[eid] for eid in ids if eid in records]),
            'matched_new': aggregate([records[eid] for eid in shared]),
            'matched_previous': aggregate([old_records[eid] for eid in shared])}
    summary_path = output / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    teacher = manifest.get('teacher_cli', 'codex')
    name = manifest.get('requested_teacher_model') or teacher
    result = {'teacher_cli': teacher, 'baseline': str(baseline),
        'requested_teacher_model': manifest.get('requested_teacher_model'),
        'requested_reasoning_effort': manifest.get('requested_reasoning_effort'),
        'baseline_model_identity': (json.loads((baseline / 'batch.json').read_text()).get('requested_teacher_model')
            or 'unpinned') if override else selection['baseline_model_identity'],
        'complete': len(records) == len(questions) and bool(summary and summary['complete']),
        'planned': len(questions), 'batch_summary': summary, 'comparability': comparability,
        'overall': aggregate(list(records.values())),
        'matched_new': aggregate([records[eid] for eid in paired]),
        'matched_previous': aggregate([old_records[eid] for eid in paired]),
        'by_family': family_results,
        'by_negative': {str(flag).lower(): aggregate([records[q['episode_id']] for q in questions
            if q['negative'] == flag and q['episode_id'] in records]) for flag in (False, True)},
        'matched_episode_ids': paired,
        'outcomes': [{'episode_id': q['episode_id'], 'family': q['family'],
            'negative': q['negative'], 'new': records.get(q['episode_id']),
            'previous': old_records.get(q['episode_id'])} for q in questions],
        'limitations': [
            'Training diagnostic: two questions per family, not held-out performance.',
            'The batch-002 teacher was not pinned; Astra High is the user identification, not verified log metadata.',
            'Requested models and efforts are recorded CLI overrides, not a server-returned model fingerprint.',
            'Evidence coverage does not prove complete search or justified absence; review before SFT.',
            'Teacher usage is only reported for episodes with usage events; missing records are not zero cost.',
            'usage_from_requests counts Claude episodes whose usage summary was missing; their input and cached '
            'tokens come from per-request records and their output is only a lower bound.',
            'Codex and Claude Code count and cache tokens differently, and their subscription allowances differ; '
            'token totals are not directly an allowance comparison.',
            'Student transcript tokens are distinct from actual teacher usage; neither measures account allowance percentage.']}
    atomic_json(output / 'comparison.json', result)

    def rate(stats):
        return f"{stats['correct_with_evidence']}/{stats['evaluated']}" if stats['evaluated'] else 'pending'

    lines = [f'# {name} {result["requested_reasoning_effort"] or ""} training comparison'.replace('  ', ' '), '',
        f"Requested teacher: `{result['requested_teacher_model']}`, reasoning `{result['requested_reasoning_effort']}`.",
        f"Evaluated {len(records)}/{len(questions)}; complete: {result['complete']}.", '',
        f"Correct with required evidence: **{rate(result['overall'])}**.",
        f"Baseline: `{result['baseline']}` ({result['baseline_model_identity']}).",
        f"On the {len(paired)} matched evaluated questions: this teacher **{rate(result['matched_new'])}**, "
        f"baseline **{rate(result['matched_previous'])}**.", '',
        '| Family | Grounded / evaluated | Matched this teacher | Matched baseline |',
        '| --- | ---: | ---: | ---: |']
    for family, stats in family_results.items():
        lines.append(f"| {family} | {rate(stats['new'])} | {rate(stats['matched_new'])} | {rate(stats['matched_previous'])} |")
    lines += ['', '| Mean per episode with usage | This teacher | Baseline |', '| --- | ---: | ---: |']
    mine, theirs = (result[key]['mean_teacher_usage_per_episode'] or {} for key in ('matched_new', 'matched_previous'))
    for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'shell_commands'):
        cell = lambda stats: f"{stats[key]:,.0f}" if key in stats else 'n/a'
        lines.append(f"| {key} (matched) | {cell(mine)} | {cell(theirs)} |")
    lines += ['', '## Interpretation', ''] + ['- ' + item for item in result['limitations']]
    lines += ['', 'Exact metrics, statuses, token-record coverage, and per-question comparisons are in `comparison.json`.', '']
    (output / 'COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps({key: result[key] for key in ('complete', 'planned', 'overall', 'matched_new', 'matched_previous')}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path,
                        help='Compare against this run instead of the selection baseline')
    args = parser.parse_args()
    report(args.output, args.baseline)


if __name__ == '__main__':
    main()
