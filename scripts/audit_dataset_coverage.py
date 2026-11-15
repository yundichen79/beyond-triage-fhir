"""Supplement the cohort audit with source hashes, coverage and checker probes."""
import copy
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys

from dataset.questions import negative, build_questions
from dataset.common import Snapshot
from scripts.audit_dataset import ROOT, OUT, IndependentChart, normalize


def statuses(value):
    if isinstance(value, dict):
        if 'status' in value:
            yield value['status']
        for child in value.values():
            yield from statuses(child)
    elif isinstance(value, list):
        for child in value:
            yield from statuses(child)


def main():
    rows = [json.loads(line) for line in (ROOT/'data/processed/cohort-200/questions.jsonl').read_text().splitlines()]
    source = ROOT/'data/raw/batch-001'
    manifest = json.loads((source/'generation.json').read_text())
    bad_hashes = []
    for name, expected in manifest['files'].items():
        path = source/name
        assert path.resolve().is_relative_to(ROOT), 'Source path leaves repository'
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            bad_hashes.append(name)
    splits = sorted({r['split'] for r in rows})
    patient_sets = {s:{r['patient_id'] for r in rows if r['split']==s} for s in splits}
    evidence_sets = {s:{ref for r in rows if r['split']==s for ref in r['ground_truth_resource_ids']} for s in splits}
    coverage = {}
    for family in sorted({r['family'] for r in rows}):
        group = [r for r in rows if r['family']==family]
        coverage[family] = {'total':len(group), 'by_split':dict(Counter(r['split'] for r in group)),
                            'negative_by_split':dict(Counter(r['split'] for r in group if negative(r['ground_truth_answer'],family))),
                            'answer_statuses':dict(Counter(s for r in group for s in statuses(r['ground_truth_answer'])))}
    # Exercise the separate implementation against the repository's miniature fixture.
    sys.path.insert(0,str(ROOT/'tests'))
    from test_dataset import fixture, config
    snapshot = Snapshot(fixture())
    fixture_rows, _, _, _ = build_questions(snapshot, config())
    probes=[]
    for row in fixture_rows:
        chart=IndependentChart(snapshot,row['patient_id'])
        answer=chart.answer(row)
        assert normalize(answer)==normalize(row['ground_truth_answer']),row['family']
        assert chart.evidence==set(row['ground_truth_resource_ids']),row['family']
        changed=copy.deepcopy(answer)
        changed['audit_corruption']=True
        assert normalize(answer)!=normalize(changed)
        assert chart.evidence!=chart.evidence|{'Observation/audit-corruption'}
        probes.append(row['family'])
    result={'manifest_files_checked':len(manifest['files']),'source_hash_mismatches':bad_hashes,
            'patient_overlap':{a+'/'+b:len(patient_sets[a]&patient_sets[b]) for a,b in combinations(splits,2)},
            'evidence_overlap':{a+'/'+b:len(evidence_sets[a]&evidence_sets[b]) for a,b in combinations(splits,2)},
            'complexity':dict(Counter(r['complexity'] for r in rows)),
            'negative_by_split':dict(Counter(r['split'] for r in rows if negative(r['ground_truth_answer'],r['family']))),
            'answer_statuses':dict(Counter(s for r in rows for s in statuses(r['ground_truth_answer']))),
            'coverage':coverage,'fixture_families_checked':sorted(set(probes)),
            'fixture_rows_and_corruption_probes_passed':len(probes)}
    (OUT/'coverage.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
