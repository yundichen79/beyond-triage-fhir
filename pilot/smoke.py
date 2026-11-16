"""CPU-only integration checks for the patient tools against isolated live HAPI."""

import argparse
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from dataset.common import file_digest, read_questions, resource_id, timestamp, write_json
from dataset.live import interrupt_cleanup, now, run_job
from pilot.contract import INTERFACE_VERSION
from pilot.tools import MAX_COUNT, PatientTools


ROOT = Path(__file__).resolve().parents[1]


def select_patient_rows(rows):
    groups = {}
    for row in rows:
        if row["split"] == "train":
            groups.setdefault(row["patient_id"], []).append(row)
    # These public task eligibilities establish the required chart coverage.
    required = {"inpatient_medications", "procedure_indications", "cbc_panel"}
    eligible = [patient for patient, group in groups.items()
                if required <= {row["family"] for row in group}]
    if not eligible:
        raise ValueError("No training patient covers the smoke test's required resource types")
    return groups[min(eligible)]


def collect(tools, kind, params, count=MAX_COUNT):
    result = tools.execute("search_fhir", {"resource_type": kind, "params": params, "count": count})
    resources = []
    pages = 0
    while True:
        if "error" in result:
            raise ValueError(result["error"])
        pages += 1
        resources.extend(result["resources"])
        if result["next_page"] is None:
            if result["complete"] is not True:
                raise ValueError("Search ended without completeness")
            return resources, pages
        result = tools.execute("next_page", {"cursor": result["next_page"]})


def codes(resource):
    return {(c.get("system"), c.get("code")) for c in resource.get("code", {}).get("coding", [])}


def point(resource, field):
    if field == "date":
        value = (resource.get("period", {}).get("start") or resource.get("effectiveDateTime")
                 or resource.get("effectivePeriod", {}).get("start") or resource.get("performedDateTime")
                 or resource.get("performedPeriod", {}).get("start"))
    else:
        value = resource.get({"authoredon": "authoredOn", "onset-date": "onsetDateTime"}[field])
    return timestamp(value)


def check_queries(client, rows, snapshot, output):
    patient = rows[0]["patient_id"]
    if {row["patient_id"] for row in rows} != {patient}:
        raise ValueError("Smoke test expects exactly one selected patient")
    tools = PatientTools(client, patient)
    client.strict_search = True
    report = {"kind": "live_filtered_query_smoke", "interface_version": INTERFACE_VERSION,
              "model_generation": False, "started_at": now(), "patient": patient,
              "source_snapshot_id": snapshot.id, "passed": False, "checks": [],
              "code_sha256": {str(p.relative_to(ROOT)): file_digest(p) for p in
                  (ROOT / "pilot/smoke.py", ROOT / "pilot/tools.py", ROOT / "pilot/contract.py",
                   ROOT / "dataset/fhir.py", ROOT / "dataset/live.py")}}

    def save():
        write_json(output / "query-smoke.json", report)

    def query(name, kind, params, expected, sort=None, count=MAX_COUNT):
        check = {"name": name, "kind": kind, "params": params, "count": count, "passed": False}
        report["checks"].append(check)
        save()
        actual, pages = collect(tools, kind, params, count)
        expected_ids = sorted(resource_id(r) for r in expected)
        actual_ids = sorted(resource_id(r) for r in actual)
        check.update(expected_ids=expected_ids, actual_ids=actual_ids, pages=pages)
        if actual_ids != expected_ids:
            raise AssertionError(f"{name}: filtered resource IDs differ from source chart")
        if sort:
            field, descending = sort
            expected_times = sorted((point(r, field) for r in expected), reverse=descending)
            actual_times = [point(r, field) for r in actual]
            check["sort_verified"] = actual_times == expected_times
            if not check["sort_verified"]:
                raise AssertionError(f"{name}: returned ordering differs from source timestamps")
        check["passed"] = True
        print(f"Query PASS: {name}, {len(actual)} resources, {pages} pages", flush=True)
        save()
        return actual

    save()
    try:
        invalid = [
            ("encounter-observation-status", "Encounter", {"status": ["final", "amended", "corrected"]}, "Invalid Encounter status"),
            ("procedure-observation-status", "Procedure", {"status": "final,amended,corrected"}, "Invalid Procedure status"),
            ("order-observation-status", "MedicationRequest", {"status": "final"}, "Invalid MedicationRequest status"),
            ("condition-observation-status", "Condition", {"clinical-status": "final"}, "Invalid Condition clinical-status"),
            ("condition-wrong-sort", "Condition", {"_sort": "-date"}, "Unsupported Condition sort"),
            ("unseen-encounter", "Observation", {"encounter": "Encounter/smoke-unseen"}, "Unseen encounter"),
            ("contradictory-statuses", "Observation", {"status": ["final", "amended"]}, "Contradictory status"),
        ]
        for name, kind, params, hint in invalid:
            before = tools.http_requests
            result = tools.execute("search_fhir", {"resource_type": kind, "params": params})
            passed = (hint in result.get("error", "") and result.get("complete") is False
                      and "resources" not in result and tools.http_requests == before)
            report["checks"].append({"name": name, "passed": passed, "error": result.get("error"),
                                     "http_requests": tools.http_requests - before})
            save()
            if not passed:
                raise AssertionError(f"{name}: invalid query was not rejected before HTTP")

        chart = {kind: list(snapshot.for_patient(patient, kind)) for kind in
                 ("Encounter", "Observation", "Condition", "Procedure", "MedicationRequest", "DiagnosticReport")}
        query("discover-encounters", "Encounter", {}, chart["Encounter"])
        stays = [r for r in chart["Encounter"] if r.get("status") == "finished" and r.get("class", {}).get("code") == "IMP"]
        if not stays:
            raise ValueError("Selected patient lacks finished inpatient encounters")
        query("finished-inpatient-sort", "Encounter", {"class": "IMP", "status": "finished", "_sort": "-date"},
              stays, sort=("date", True), count=1)
        target = min(stays, key=lambda r: r["id"])
        reference = resource_id(target)
        linked = [r for r in chart["Condition"] if r.get("encounter", {}).get("reference") == reference]
        query("encounter-reference", "Condition", {"encounter": reference}, linked)

        dated_conditions = [r for r in chart["Condition"] if r.get("onsetDateTime")
                            and r.get("clinicalStatus", {}).get("coding")]
        if not dated_conditions:
            raise ValueError("Selected patient lacks dated conditions with clinical status")
        clinical_status = dated_conditions[0]["clinicalStatus"]["coding"][0]["code"]
        conditions = [r for r in dated_conditions if any(
            c.get("code") == clinical_status for c in r["clinicalStatus"]["coding"])]
        lower = min(point(r, "onset-date") for r in conditions)
        upper = max(point(r, "onset-date") for r in conditions) + timedelta(seconds=1)
        query("condition-clinical-status-onset-sort", "Condition", {
            "clinical-status": clinical_status,
            "onset-date": ["ge" + lower.isoformat(), "lt" + upper.isoformat()],
            "_sort": "onset-date"}, conditions, sort=("onset-date", False))

        observations = [r for r in chart["Observation"] if r.get("status") in {"final", "amended", "corrected"}]
        candidates = sorted({c for r in observations for c in codes(r) if all(c)})
        preferred = [("http://loinc.org", "2160-0"), ("http://loinc.org", "718-7")]
        selected = None
        for code in preferred + candidates:
            matching = [r for r in observations if code in codes(r)]
            if matching and all(r.get("effectiveDateTime") for r in matching):
                selected, observations = code, matching
                break
        if selected is None:
            raise ValueError("No point-valued observation code available for date checks")
        lower = min(point(r, "date") for r in observations)
        upper = max(point(r, "date") for r in observations) + timedelta(seconds=1)
        params = {"code": "|".join(selected), "status": "final,amended,corrected",
                  "date": ["ge" + lower.isoformat(), "lt" + upper.isoformat()], "_sort": "date"}
        found = query("observation-code-status-date-and-sort", "Observation", params, observations,
                      sort=("date", False), count=2)
        strict_params = dict(params, date=["gt" + lower.isoformat(), "lt" + upper.isoformat()])
        query("exclusive-lower-date-bound", "Observation", strict_params,
              [r for r in observations if point(r, "date") > lower], sort=("date", False))
        result = tools.execute("read_fhir", {"reference": resource_id(found[0])})
        passed = "error" not in result and resource_id(result["resource"]) == resource_id(found[0])
        report["checks"].append({"name": "read-discovered-observation", "passed": passed})
        if not passed:
            raise AssertionError("Read of a discovered observation failed")

        reports = [r for r in chart["DiagnosticReport"] if ("http://loinc.org", "58410-2") in codes(r)
                   and r.get("status") in {"final", "amended", "corrected"}]
        if not reports:
            raise ValueError("Selected patient lacks CBC reports")
        query("cbc-report-status-code-sort", "DiagnosticReport", {
            "code": "http://loinc.org|58410-2", "status": "final,amended,corrected", "_sort": "-date"},
            reports, sort=("date", True), count=2)

        procedures = [r for r in chart["Procedure"] if r.get("status") == "completed"]
        if not procedures:
            raise ValueError("Selected patient lacks completed procedures")
        query("completed-procedures", "Procedure", {"status": "completed", "_sort": "date"},
              procedures, sort=("date", False))

        orders = [r for r in chart["MedicationRequest"] if r.get("status") == "completed" and r.get("authoredOn")]
        if not orders:
            raise ValueError("Selected patient lacks dated completed medication orders")
        lower = min(point(r, "authoredon") for r in orders)
        upper = max(point(r, "authoredon") for r in orders) + timedelta(seconds=1)
        query("completed-order-authoredon-and-sort", "MedicationRequest", {
            "status": "completed", "authoredon": ["ge" + lower.isoformat(), "lt" + upper.isoformat()],
            "_sort": "authoredon"}, orders, sort=("authoredon", False))

        empty_id = "smoke-confirmed-nonexistent-id"
        if any(r["id"] == empty_id for r in chart["Observation"]):
            raise ValueError("Nonexistent-ID sentinel unexpectedly exists")
        query("valid-empty-query", "Observation", {"_id": empty_id}, [])
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report.update(finished_at=now(), checks_passed=sum(c["passed"] for c in report["checks"]),
                      checks_total=len(report["checks"]), http_requests=tools.http_requests)
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/cohort-200")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = select_patient_rows(read_questions(args.dataset / "questions.jsonl"))
    live_args = SimpleNamespace(output=args.output, dataset=args.dataset, bundles=None,
                                snapshot=None, batch_size=100,
                                callback_label="filtered query smoke tests", callback_status="running_query_smoke")
    def after_verify(client, selected, snapshot):
        check_queries(client, selected, snapshot, args.output)
    with interrupt_cleanup():
        run_job(live_args, rows=rows, after_verify=after_verify)


if __name__ == "__main__":
    main()
