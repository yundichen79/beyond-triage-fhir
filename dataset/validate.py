"""Recompute clinical answers and check question definitions, evidence and splits."""

import argparse
import sqlite3
import json
from collections import Counter
from pathlib import Path

from dataset.common import QUESTION_FIELDS, digest, patient_ref, read_questions
from dataset.oracle import evaluate
from dataset.questions import definitions, split_for
from dataset.snapshot import add_source_arguments, open_records

ROOT = Path(__file__).resolve().parents[1]


def validate_rows(snapshot, rows, splits, config):
    errors = []
    if not rows:
        errors.append("Dataset contains no questions")
    if not isinstance(splits, dict):
        raise ValueError("patient_splits.json must map patient IDs to split names")
    if {r["patient_id"] for r in rows} - splits.keys():
        errors.append("Questions include patients without split assignments")
    for patient, split in splits.items():
        if patient not in snapshot.patients:
            errors.append(f"Unknown patient in splits: {patient}")
        if split != split_for(patient, config["split_seed"], config["split_ratios"]):
            errors.append(f"Invalid split assignment: {patient}")
    for label, counts in (("question ID", Counter(r["id"] for r in rows)),
                          ("patient/family", Counter((r["patient_id"], r["family"]) for r in rows))):
        errors.extend(f"Duplicate {label}: {key}" for key, count in counts.items() if count != 1)
    canonical_definitions = {}
    for row in rows:
        try:
            patient = row["patient_id"]
            if set(row) != set(QUESTION_FIELDS):
                raise ValueError("Question schema mismatch")
            if row["split"] != splits[patient]:
                raise ValueError("Patient split leakage")
            if patient not in canonical_definitions:
                if len(splits) > 20 and len(canonical_definitions) % 20 == 0:
                    print(f"Validating clinical answers for patient {len(canonical_definitions) + 1}/{len(splits)}", flush=True)
                canonical_definitions[patient] = {d[0]: d for d in definitions(snapshot, patient, config)}
            _, complexity, operations, text, spec = canonical_definitions[patient][row["family"]]
            if (row["question"], row["oracle_spec"], row["complexity"], row["required_operations"], row["answer_type"]) != (
                    text, spec, complexity, operations, "object"):
                raise ValueError("Question text or computation rules differ from canonical definition")
            if row["id"] != row["family"] + "_" + digest([patient, spec])[:20]:
                raise ValueError("Question ID mismatch")
            answer, evidence = evaluate(snapshot, patient, spec)
            if answer != row["ground_truth_answer"]:
                raise ValueError("Answer mismatch")
            if evidence != sorted(row["ground_truth_resource_ids"]):
                raise ValueError("Supporting resource IDs mismatch")
            for ref in evidence:
                owner = patient_ref(snapshot.resources[ref])
                if owner and owner != patient:
                    raise ValueError(f"Evidence belongs to another patient: {ref}")
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            errors.append(f"{row.get('id')}: {exc}")
    return {"valid": not errors, "questions_checked": len(rows), "errors": errors}


def validate(snapshot, directory, config=None):
    config = config if config is not None else json.loads((ROOT / "configs/cohort.json").read_text())
    return validate_rows(snapshot, read_questions(directory / "questions.jsonl"),
                         json.loads((directory / "patient_splits.json").read_text()), config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cohort.json")
    args = parser.parse_args()
    try:
        with open_records(args.bundles, args.snapshot) as records:
            result = validate(records, args.dataset, json.loads(args.config.read_text()))
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"Checked {result['questions_checked']} questions; {len(result['errors'])} errors.")
    if not result["valid"]:
        parser.exit(1, "\n".join(result["errors"]) + "\n")


if __name__ == "__main__":
    main()
