"""Replay labels from checksummed raw bundles without creating review exports.

This independently reloads the source data, but uses the same answer rules as the
generator. It is a source-replay check, not an independent oracle or human review.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from dataset.bundles import iter_bundle_entries
from dataset.common import Snapshot, file_digest, read_questions, resource_id
from dataset.oracle import PATIENT_TYPES, evaluate
from dataset.snapshot import DEFAULT_BUNDLES, walk_references


def source_snapshot(path):
    # Billing documents and narrative attachments are irrelevant to these tasks
    # and can otherwise dominate memory on the largest Synthea chart.
    aliases, resources = {}, []
    for entry in iter_bundle_entries(path):
        resource = entry["resource"]
        if resource["resourceType"] not in PATIENT_TYPES | {"Patient", "Medication"}:
            continue
        if entry.get("fullUrl"):
            aliases[entry["fullUrl"]] = resource_id(resource)
        resources.append(resource)
    for resource in resources:
        for container, ref in walk_references(resource):
            if ref in aliases:
                container["reference"] = aliases[ref]
    return Snapshot(resources)


def review_questions(manifest, rows, bundles):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["patient_id"]].append(row)
    errors, checked = [], 0
    for index, (patient, group) in enumerate(grouped.items(), 1):
        directories = [bundles] if isinstance(bundles, Path) else bundles
        matches = [directory / (patient.split("/", 1)[1] + ".json") for directory in directories
                   if (directory / (patient.split("/", 1)[1] + ".json")).is_file()]
        if len(matches) != 1:
            raise ValueError(f"Expected one original patient bundle for {patient}")
        path = matches[0]
        if manifest["source_files"].get(str(path.resolve())) != file_digest(path):
            raise ValueError(f"Raw bundle differs from preserved source: {path}")
        snapshot = source_snapshot(path)
        for row in group:
            try:
                answer, evidence = evaluate(snapshot, patient, row["oracle_spec"])
                if answer != row["ground_truth_answer"] or evidence != sorted(row["ground_truth_resource_ids"]):
                    raise ValueError("Raw source answer/evidence mismatch")
                checked += 1
            except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
                errors.append({"id": row["id"], "error": str(exc)})
        # Release the chart before loading the next patient.
        del snapshot
        if index % 20 == 0:
            print(f"Replayed raw source for {index}/{len(grouped)} patients", flush=True)
    return {"passed": bool(rows) and not errors, "questions_checked": checked, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, nargs="+", default=[DEFAULT_BUNDLES])
    args = parser.parse_args()
    try:
        manifest = {"source_files": {}}
        for directory in args.bundles:
            generation = json.loads((directory.parent / "generation.json").read_text())
            if generation.get("status") != "complete":
                raise ValueError(f"Generation incomplete: {directory}")
            expected = {name: checksum for name, checksum in generation["files"].items()
                        if name.startswith("fhir/") and name.endswith(".json")}
            actual = {str(path.relative_to(directory.parent)) for path in directory.glob("*.json")}
            if set(expected) != actual:
                raise ValueError(f"Missing/extra source bundles: {directory}")
            manifest["source_files"].update({str((directory.parent / name).resolve()): checksum
                                             for name, checksum in expected.items()})
        result = review_questions(manifest, read_questions(args.dataset / "questions.jsonl"), args.bundles)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"Source replay: {result['questions_checked']} questions; passed={result['passed']}")
    if not result["passed"]:
        parser.exit(1, json.dumps(result["errors"], indent=2) + "\n")


if __name__ == "__main__":
    main()
