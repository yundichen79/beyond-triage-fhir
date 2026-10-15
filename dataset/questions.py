"""Build clinical chart-review QA and patient splits; keep only two output files."""

import argparse
import sqlite3
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dataset.common import digest, write_json, write_questions
from dataset.oracle import InvalidExample, IneligibleExample, evaluate
from dataset.snapshot import add_source_arguments, open_records

ROOT = Path(__file__).resolve().parents[1]
# Difficulty describes intended reasoning, pending calibration with agent rollouts.
TASKS = {
    "inpatient_conditions": ("simple", ["encounter_selection", "condition_join"],
        "For the most recent completed inpatient stay, list its linked condition codes, names, and recorded onset dates."),
    "inpatient_medications": ("simple", ["encounter_selection", "medication_resolution"],
        "For the most recent completed inpatient stay with at least one recorded non-draft, non-error medication order, list linked medication orders: coded product, authored date, and current recorded order status. Orders do not establish administration."),
    "cbc_panel": ("simple", ["report_selection", "result_reference_traversal"],
        "Review the latest final, amended, or corrected CBC report (LOINC 58410-2). Give its report time and the hemoglobin (718-7, g/dL) and platelet (777-3, 10*3/uL) results referenced by that report, including measurement times. Do not substitute unrelated observations."),
    "outpatient_bp": ("simple", ["encounter_selection", "component_extraction"],
        "For the most recent completed ambulatory encounter (class AMB) with a linked blood-pressure observation, report the most recent final, amended, or corrected systolic and diastolic blood-pressure measurements linked to that encounter, with values in mmHg and measurement times. Report a missing record or missing numeric value explicitly."),
    "preadmission_treated_conditions": ("intermediate", ["encounter_selection", "condition_timeline", "prescribing_time_window", "reason_reference_traversal", "medication_resolution"],
        "Review treatment history before the most recent completed inpatient stay. Which conditions had recorded onset before admission, no recorded abatement before admission, and an explicit Condition reasonReference from a non-draft, non-error medication order authored in the preceding 365 days, including the lower boundary and excluding admission itself? Give each condition's code, name, onset and abatement dates, and the linked coded medication products and order dates. Report no matching documented treatment history if none exists. These records do not establish medication use or active disease at admission."),
    "discharge_followup": ("intermediate", ["discharge_anchor", "time_window", "condition_join"],
        "For the most recent completed inpatient stay with a full 30-day window before the record cutoff, identify the first completed ambulatory encounter strictly after discharge and no later than 30 days afterward. Give its time, elapsed days rounded to one decimal, and linked conditions; otherwise report no matching follow-up."),
    "admission_medication_comparison": ("intermediate", ["two_encounter_selection", "medication_resolution", "set_comparison"],
        "Compare coded medication products ordered during the two most recent completed inpatient stays. Report products in both, only the earlier stay, and only the later stay. Match by coding system and product code; absence does not establish discontinuation."),
    "procedure_indications": ("intermediate", ["procedure_selection", "encounter_traversal", "reason_reference_traversal"],
        "Find the most recent completed encounter containing completed procedures with explicit Condition reasonReference links. List those procedures with codes, names, performed times, and their linked condition indications. Exclude procedures without explicit Condition indication references; do not infer indications from procedure names or conditions elsewhere in the encounter."),
    "medication_indications": ("intermediate", ["prescribing_encounter_selection", "medication_resolution", "reason_reference_traversal"],
        "Find the most recent completed encounter containing medication orders with explicit Condition reasonReference links. List those coded medications and their linked condition indications. Exclude orders without explicit indication references; do not infer indications from drug names."),
    "cbc_change": ("intermediate", ["paired_report_selection", "time_window", "result_reference_traversal", "comparison"],
        "Review a short-interval CBC trend. Find the most recent final, amended, or corrected CBC report (LOINC 58410-2) that has an earlier qualifying CBC report within the preceding 90 days, including the lower boundary and excluding the same timestamp. Compare it with the most recent of those earlier reports. For each, give referenced hemoglobin (718-7, g/dL) and platelet (777-3, 10*3/uL) results and dates, and the later-minus-earlier changes rounded to one decimal. Do not substitute reports farther apart or results from other reports; state any missing record or numeric value."),
    "glycemic_followup": ("complex", ["condition_anchor", "time_windows", "conditional_retrieval", "comparison"],
        "Review documented first-year glycemic testing after the first recorded onset of type 2 diabetes (SNOMED 44054006) or prediabetes (714628002). Was HbA1c (LOINC 4548-4) measured strictly afterward and before day 365? If so, give the first result and compare it with the latest result in the 90 days up to and including onset. Include dates, values in percent, and change in percentage points rounded to one decimal. Require the full follow-up window before the record cutoff. Distinguish missing follow-up, missing baseline, and missing numeric values; do not infer treatment effect or whether care met a guideline."),
    "renal_followup": ("complex", ["measurement_anchor", "encounter_traversal", "time_window", "conditional_retrieval", "comparison"],
        "Review documented first-year renal laboratory follow-up, starting with the first recorded final, amended, or corrected creatinine measurement (LOINC 2160-0) linked to a completed ambulatory encounter. Was creatinine remeasured strictly afterward and before day 365? If so, compare the first subsequent result with that baseline. Give dates, values in mg/dL, and change rounded to one decimal. Require the full follow-up window before the record cutoff, and distinguish no follow-up from missing numeric values. This is a record review, not a judgment about recommended monitoring intervals."),
    "followup_medication_comparison": ("complex", ["discharge_anchor", "conditional_encounter_search", "medication_resolution", "set_comparison"],
        "For the most recent completed inpatient stay that contains at least one recorded non-draft, non-error medication order and has a full 30-day window before the record cutoff, find the first completed ambulatory encounter strictly after discharge and no later than day 30. If one exists, compare coded medication products ordered in the stay and at follow-up: both, inpatient only, and follow-up only. Match by coding system and product code. Otherwise report no matching follow-up. Compare encounter-linked orders; these are not discharge prescriptions or a reconciled list of medications being taken, and missing orders do not establish discontinuation."),
    "subsequent_admission": ("complex", ["admission_anchor", "conditional_encounter_search", "condition_join", "set_comparison"],
        "After the first recorded completed inpatient stay, identify the next completed inpatient admission strictly after its discharge and before the record cutoff. If one exists, give its admission time, the elapsed days rounded to one decimal, its linked conditions, and condition codes shared by the two stays. Otherwise report no subsequent recorded admission."),
}
FAMILIES = set(TASKS)


def split_for(patient, seed, ratios):
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Expected three positive split ratios summing to 1")
    score = int(digest([seed, patient]), 16) / 2**256
    return "train" if score < ratios[0] else "validation" if score < sum(ratios[:2]) else "test"


def definitions(snapshot, patient, config):
    cutoff = (datetime.strptime(config["end_date"], "%Y%m%d").replace(tzinfo=timezone.utc)
              + timedelta(days=1)).isoformat()
    for family, (complexity, operations, text) in TASKS.items():
        spec = {"operation": family, "as_of": cutoff}
        if family in {"discharge_followup", "followup_medication_comparison"}:
            days = (30, 60, 90)[int(digest([patient, family, "followup_window"]), 16) % 3]
            spec["followup_days"] = days
            text = text.replace("30-day", f"{days}-day").replace("30 days", f"{days} days").replace("day 30", f"day {days}")
        if family in {"renal_followup", "glycemic_followup"}:
            spec["followup_days"] = 365
        if family == "cbc_change":
            spec["comparison_days"] = 90
        if family == "preadmission_treated_conditions":
            spec["lookback_days"] = 365
        if family == "followup_medication_comparison":
            spec["inpatient_orders_required"] = True
        yield family, complexity, operations, (
            f"Patient {patient.split('/', 1)[1]}. {text} Use records before {cutoff}; "
            "use UTC times and only final, amended, or corrected observations."
        ), spec


def negative(answer, family):
    if family in {"discharge_followup", "followup_medication_comparison", "renal_followup", "glycemic_followup"}:
        followup = answer.get("followup")
        return followup is None or followup.get("status") == "no_matching_record"
    if family == "subsequent_admission":
        return answer.get("subsequent_admission") is None
    if family == "inpatient_conditions":
        return not answer["conditions"]
    if family in {"inpatient_medications", "medication_indications"}:
        return not answer["orders"]
    if family == "procedure_indications":
        return not answer["procedures"]
    if family == "preadmission_treated_conditions":
        return not answer["conditions_with_prior_orders"]
    if family == "outpatient_bp":
        return all(answer[k].get("status") == "no_matching_record" for k in ("systolic", "diastolic"))
    if family == "admission_medication_comparison":
        return not any(answer["products"].values())
    if family == "cbc_change":
        return all(answer[period][key].get("status") == "no_matching_record"
                   for period in ("previous", "latest") for key in ("hemoglobin", "platelets"))
    if family == "cbc_panel":
        return all(answer[k].get("status") == "no_matching_record" for k in ("hemoglobin", "platelets"))
    return False


def build_questions(snapshot, config, max_patients=None):
    patients = sorted(snapshot.patients, key=lambda p: digest([config["split_seed"], "pilot", p]))
    if max_patients is not None:
        patients = patients[:max_patients]
    splits = {p: split_for(p, config["split_seed"], config["split_ratios"]) for p in patients}
    rows, rejected = [], []
    for index, patient in enumerate(patients, 1):
        for family, complexity, operations, question, spec in definitions(snapshot, patient, config):
            try:
                answer, evidence = evaluate(snapshot, patient, spec)
            except IneligibleExample as exc:
                rejected.append({"patient_id": patient, "family": family, "kind": "ineligible", "reason": str(exc)})
                continue
            except (InvalidExample, ValueError, TypeError, KeyError, ArithmeticError) as exc:
                rejected.append({"patient_id": patient, "family": family, "kind": "invalid", "reason": str(exc)})
                continue
            rows.append({"id": family + "_" + digest([patient, spec])[:20], "patient_id": patient,
                         "question": question, "family": family, "complexity": complexity,
                         "required_operations": operations, "split": splits[patient],
                         "ground_truth_answer": answer, "ground_truth_resource_ids": evidence,
                         "answer_type": "object", "oracle_spec": spec})
        if len(patients) > 20 and index % 20 == 0:
            print(f"Built clinical candidates for {index}/{len(patients)} patients", flush=True)
    # Retain representative true negatives without an all-empty family dominating training.
    # Sampling is independent of split labels and stable by patient/question ID.
    removed = set()
    coverage = {}
    for family in sorted(FAMILIES):
        group = [r for r in rows if r["family"] == family]
        positives = [r for r in group if not negative(r["ground_truth_answer"], family)]
        negatives = sorted((r for r in group if negative(r["ground_truth_answer"], family)), key=lambda r: digest(r["id"]))
        limit = max(1, len(positives) // 3) if positives else 0
        removed.update(r["id"] for r in negatives[limit:])
        coverage[family] = {"positive": len(positives), "negative": min(len(negatives), limit),
                            "eligible": len(group), "accepted": len(positives) + min(len(negatives), limit)}
    rows = [r for r in rows if r["id"] not in removed]
    report = {"patients": len(patients), "patients_with_questions": len({r["patient_id"] for r in rows}),
              "questions": len(rows), "families": coverage, "rejections": dict(Counter(r["kind"] for r in rejected)),
              "patient_splits": dict(Counter(splits.values())), "question_splits": dict(Counter(r["split"] for r in rows)),
              "complexity": dict(Counter(r["complexity"] for r in rows)),
              "missing_families_by_split": {s: sorted(FAMILIES - {r["family"] for r in rows if r["split"] == s})
                                            for s in ("train", "validation", "test")}}
    return rows, rejected, splits, report


def save_dataset(output, rows, splits, replace=False):
    if output.exists() and not replace:
        raise ValueError(f"Dataset exists: {output}; use --replace to update the current dataset.")
    if output.exists() and any(p.name not in {"questions.jsonl", "patient_splits.json"} for p in output.iterdir()):
        raise ValueError("Refusing to replace a directory containing unrelated files")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".qa-build-", dir=output.parent))
    try:
        write_questions(temporary / "questions.jsonl", rows)
        write_json(temporary / "patient_splits.json", splits)
        output.mkdir(exist_ok=True)
        for name in ("questions.jsonl", "patient_splits.json"):
            (temporary / name).replace(output / name)
    finally:
        shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cohort.json")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.max_patients is not None and args.max_patients <= 0:
        parser.error("--max-patients must be positive")
    try:
        config = json.loads(args.config.read_text())
        with open_records(args.bundles, args.snapshot) as snapshot:
            if args.max_patients and args.max_patients > len(snapshot.patients):
                raise ValueError("Requested more patients than the snapshot contains")
            rows, rejected, splits, report = build_questions(snapshot, config, args.max_patients)
            from dataset.validate import validate_rows
            check = validate_rows(snapshot, rows, splits, config)
            if not check["valid"]:
                raise ValueError("Generated questions failed validation: " + str(check["errors"][:5]))
            save_dataset(args.output, rows, splits, replace=args.replace)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(report, indent=2))
    invalid = Counter((r["family"], r["reason"]) for r in rejected if r["kind"] == "invalid")
    if invalid:
        print("Rejected invalid candidates:", dict(invalid))


if __name__ == "__main__":
    main()
