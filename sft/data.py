"""Turn the selected teacher trajectories into SFT examples in the student's exact format.

Cleaning: failed actions are removed before rendering, so the student is never trained
on a mistake. A malformed assistant turn is dropped with its format-error feedback. A
tool call whose result is an error is dropped with that result, and an assistant turn
left with no calls is dropped. Failed actions retrieve nothing, so every remaining
action and result is real and in its original order.

Modes: `clean` removes failed actions as above. `recovery` keeps the conversation exactly
as it happened, failed actions and error feedback included, but never trains on a failed
turn (malformed, or containing a failed call); the corrected turns that follow are trained
in the context of the failure, which teaches recovery.

Examples: one per (trained) assistant turn. `prompt` is the text the student sees before the turn,
rendered with the pinned Qwen chat template and thinking disabled, as at inference.
`completion` is the turn in Qwen's native tool-call format, ending with <|im_end|>.
Rendering the prompt with the generation prompt and the conversation through that turn
must agree on the prompt text; any mismatch stops the conversion.

Split: about 10% of patients form a dev set for monitoring training loss. The dataset's
validation split remains for choosing settings by agent runs.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from dataset.common import digest
from student.agent import MODEL_CACHE, MODEL_ID, MODEL_REVISION, Settings
from teacher.bridge import atomic_json

DEV_FRACTION = 0.1
DEV_SEED = 'sft-dev-v1'


def is_error(message):
    if message.get('role') != 'tool':
        return False
    try:
        result = json.loads(message['content'])
    except (TypeError, ValueError):
        return False
    return isinstance(result, dict) and 'error' in result


def tool_results(messages, index):
    """The tool messages answering the assistant turn at `index`, in call order."""
    calls, results = messages[index].get('tool_calls') or [], []
    while (len(results) < len(calls) and index + 1 + len(results) < len(messages)
           and messages[index + 1 + len(results)]['role'] == 'tool'):
        results.append(messages[index + 1 + len(results)])
    return results


def failed_turns(messages):
    """Indices of assistant turns that were malformed or contained a failed tool call."""
    failed = set()
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        following = messages[index + 1] if index + 1 < len(messages) else None
        if (not message.get('tool_calls') and following and following['role'] == 'user'
                and following['content'].startswith('Output format error')):
            failed.add(index)
        elif any(is_error(result) for result in tool_results(messages, index)):
            failed.add(index)
    return failed


def clean(messages):
    """Remove failed actions; return (cleaned messages, removal counts)."""
    cleaned, removed, index = [], Counter(), 0
    while index < len(messages):
        message = messages[index]
        following = messages[index + 1] if index + 1 < len(messages) else None
        if (message['role'] == 'assistant' and not message.get('tool_calls') and following
                and following['role'] == 'user' and following['content'].startswith('Output format error')):
            removed['malformed_turns'] += 1
            index += 2
            continue
        if message['role'] == 'assistant' and message.get('tool_calls'):
            calls = message['tool_calls']
            results = []
            while (len(results) < len(calls) and index + 1 + len(results) < len(messages)
                   and messages[index + 1 + len(results)]['role'] == 'tool'):
                results.append(messages[index + 1 + len(results)])
            kept_calls, kept_results = [], []
            for position, call in enumerate(calls):
                # A call without a result is the final finish, which ended the episode.
                result = results[position] if position < len(results) else None
                if result is not None and is_error(result):
                    removed['failed_calls'] += 1
                    continue
                kept_calls.append(call)
                if result is not None:
                    kept_results.append(result)
            index += 1 + len(results)
            if not kept_calls:
                removed['empty_turns'] += 1
                continue
            cleaned.append(dict(message, tool_calls=kept_calls))
            cleaned.extend(kept_results)
            continue
        cleaned.append(message)
        index += 1
    return cleaned, removed


def render(tokenizer, messages, tools, generation_prompt):
    return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                         add_generation_prompt=generation_prompt, enable_thinking=False)


def examples(record, system_prompt, tools, tokenizer, max_prompt_tokens, mode='clean'):
    """Per-turn (prompt, completion) examples for one trajectory."""
    if mode == 'clean':
        messages, removed = clean(record['messages'])
        skipped = set()
    elif mode == 'recovery':
        messages = record['messages']
        # Offsets shift by one for the system message added below.
        skipped = {index + 1 for index in failed_turns(messages)}
        removed = Counter(untrained_failed_turns=len(skipped)) if skipped else Counter()
    else:
        raise ValueError(f'Unknown mode {mode}')
    conversation = [{'role': 'system', 'content': system_prompt}] + messages
    if conversation[-1]['role'] != 'assistant' or conversation[-1]['tool_calls'][-1]['function']['name'] != 'finish':
        raise ValueError(f"{record['episode_id']}: trajectory does not end with finish")
    result = []
    for position, message in enumerate(conversation):
        if message['role'] != 'assistant' or position in skipped:
            continue
        prompt = render(tokenizer, conversation[:position], tools, True)
        full = render(tokenizer, conversation[:position + 1], tools, False)
        if not full.startswith(prompt):
            raise ValueError(f"{record['episode_id']} turn {len(result)}: prompt is not a prefix of the rendered turn")
        completion = full[len(prompt):]
        if not completion.rstrip('\n').endswith('<|im_end|>'):
            raise ValueError(f"{record['episode_id']} turn {len(result)}: completion does not end the turn")
        completion = completion.rstrip('\n')
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_tokens > max_prompt_tokens:
            raise ValueError(f"{record['episode_id']}: prompt exceeds the student context budget")
        result.append({'id': f"{record['episode_id']}-t{len(result)}", 'episode_id': record['episode_id'],
                       'conversation_index': position,
                       'patient_id': record['patient_id'], 'family': record['family'],
                       'complexity': record['complexity'], 'turn': len(result), 'prompt': prompt,
                       'completion': completion, 'prompt_tokens': prompt_tokens,
                       'completion_tokens': len(tokenizer.encode(completion, add_special_tokens=False))})
    return result, removed


def dev_patient(patient):
    return int(digest([DEV_SEED, patient])[:8], 16) / 0xFFFFFFFF < DEV_FRACTION


def build(corpus, output, tokenizer, mode='clean'):
    if output.exists():
        raise ValueError(f'Refusing to replace existing SFT data: {output}')
    meta = json.loads((corpus / 'corpus.json').read_text())
    records = [json.loads(line) for line in (corpus / 'trajectories.jsonl').read_text().splitlines()]
    selected = [r for r in records if r['selected']]
    settings = Settings()
    budget = settings.max_context_tokens - settings.max_new_tokens
    splits, removed, trajectories = {'train': [], 'dev': []}, Counter(), Counter()
    for record in selected:
        rows, counts = examples(record, meta['system_prompt'], meta['tools'], tokenizer, budget, mode)
        removed.update(counts)
        trajectories['with_failed_turns' if counts else 'unchanged'] += 1
        split = 'dev' if dev_patient(record['patient_id']) else 'train'
        trajectories[split] += 1
        splits[split].extend(rows)
    output.mkdir(parents=True, mode=0o700)
    for split, rows in splits.items():
        with (output / f'{split}.jsonl').open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = {split: {'trajectories': trajectories[split], 'examples': len(rows),
                       'patients': len({r['patient_id'] for r in rows}),
                       'prompt_tokens': sum(r['prompt_tokens'] for r in rows),
                       'completion_tokens': sum(r['completion_tokens'] for r in rows),
                       'max_prompt_tokens': max((r['prompt_tokens'] for r in rows), default=0)}
               for split, rows in splits.items()}
    atomic_json(output / 'sft-data.json', {
        'corpus': str(corpus), 'corpus_dataset_sha256': meta['dataset_sha256'],
        'student_interface_sha256': meta['student_interface_sha256'],
        'tokenizer': {'model': MODEL_ID, 'revision': MODEL_REVISION},
        'format': 'One example per assistant turn: prompt rendered with add_generation_prompt=True and '
                  'enable_thinking=False; completion is the turn in Qwen tool-call format ending with <|im_end|>. '
                  'Train with loss on the completion only.',
        'mode': mode,
        'cleaning': ('Malformed turns and failed tool calls removed with their error feedback.' if mode == 'clean' else
                     'Conversation kept as recorded; failed turns stay in context but are never trained on.'),
        'dev_split': f'{DEV_FRACTION:.0%} of patients by hash ({DEV_SEED}); the dataset validation split is unused.',
        'selected_trajectories': len(selected), 'trajectories': dict(trajectories), 'removed': dict(removed),
        'splits': summary})
    return summary, dict(removed), dict(trajectories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True, help='Folder written by teacher.collect')
    parser.add_argument('--output', type=Path, required=True, help='New SFT data folder')
    parser.add_argument('--mode', choices=('clean', 'recovery'), default='clean',
                        help='clean: remove failed actions; recovery: keep them in context, never train on them')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(MODEL_CACHE),
                                              local_files_only=True, trust_remote_code=False)
    print(json.dumps(build(args.corpus, args.output, tokenizer, args.mode), indent=2))


if __name__ == '__main__':
    main()
