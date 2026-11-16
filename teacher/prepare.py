"""Export public training inputs without copying labels, reports, or source code."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

from pilot.agent import MODEL_REVISION, SYSTEM_PROMPT, Settings
from pilot.contract import INTERFACE_VERSION, contract_for_question, json_schema
from pilot.tools import tool_definitions

ROOT = Path(__file__).resolve().parents[1]


def opaque_id(row):
    return hashlib.sha256(('teacher-v1:' + row['id']).encode()).hexdigest()[:24]


def public_episode(row):
    if row['split'] != 'train':
        raise ValueError('Only training rows may be exported')
    _, shape = contract_for_question(row['question'])
    schema = json_schema(shape)
    return {
        'episode_id': opaque_id(row), 'question': row['question'],
        'patient_reference': row['patient_id'], 'interface_version': INTERFACE_VERSION,
        'system_prompt': SYSTEM_PROMPT, 'answer_schema': schema,
        'tools': tool_definitions(),
        'budgets': {key: value for key, value in asdict(Settings()).items()
                    if key.startswith('max_')},
        'token_accounting': {
            'tokenizer': 'Qwen/Qwen3-8B', 'revision': MODEL_REVISION,
            'scope': 'Rendered student transcript and submitted assistant text only; '
                     'does not measure or constrain Codex hidden reasoning or outer chat context.',
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare(dataset, output):
    if output.exists():
        raise ValueError(f'Refusing to replace an existing public export: {output}')
    rows = [json.loads(line) for line in (dataset / 'questions.jsonl').read_text().splitlines()
            if line.strip()]
    packets = [public_episode(row) for row in rows if row['split'] == 'train']
    ids = [p['episode_id'] for p in packets]
    if not packets or len(ids) != len(set(ids)):
        raise ValueError('Empty training selection or duplicate episode IDs')
    output.mkdir(parents=True)
    (output / 'episodes').mkdir()
    for packet in packets:
        write_json(output / 'episodes' / (packet['episode_id'] + '.json'), packet)
    write_json(output / 'index.json', {'interface_version': INTERFACE_VERSION,
                                     'episode_ids': sorted(ids), 'count': len(ids)})
    shutil.copyfile(ROOT / 'teacher_workspace/TEACHER.md', output / 'TEACHER.md')
    print(f'Exported {len(packets)} training-only public packets to {output}')
    return packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/processed/cohort-200')
    parser.add_argument('--output', type=Path, default=ROOT / 'teacher_workspace/public')
    args = parser.parse_args()
    prepare(args.dataset, args.output)


if __name__ == '__main__':
    main()
