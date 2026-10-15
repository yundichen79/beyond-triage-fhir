"""Import snapshots with stable IDs; verify evidence against a live FHIR R4 server."""

import argparse
import sqlite3
import json
from itertools import islice
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dataset.common import Snapshot, canonical, patient_ref, read_questions, write_json
from dataset.oracle import PATIENT_TYPES, evaluate
from dataset.snapshot import add_source_arguments, open_records


class FHIR:
    def __init__(self, base_url, timeout=180, strict_search=False):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.strict_search = strict_search
        if urlparse(self.base).scheme not in {"http", "https"}:
            raise ValueError("--base-url must be HTTP(S)")

    def request(self, path="", body=None):
        url = path if path.startswith(("http://", "https://")) else f"{self.base}/{path.lstrip('/')}"
        # Do not follow pagination links to unrelated hosts/servers.
        parsed, base = urlparse(url), urlparse(self.base)
        if (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc) or not (
                parsed.path == base.path or parsed.path.startswith(base.path.rstrip("/") + "/")):
            raise ValueError(f"FHIR URL is outside the configured server: {url}")
        headers = {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json",
                   "Cache-Control": "no-cache"}
        if self.strict_search:
            headers["Prefer"] = "handling=strict"
        request = Request(url, data=canonical(body).encode() if body is not None else None, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.load(response)
        except HTTPError as exc:
            detail = exc.read(8192).decode("utf-8", errors="replace")
            raise ValueError(f"FHIR HTTP {exc.code} for {url}: {detail}") from exc
        if value.get("resourceType") == "OperationOutcome":
            raise ValueError(f"FHIR OperationOutcome: {value.get('issue')}")
        return value

    def search_all(self, resource_type, patient=None):
        params = {"_count": 200, "_total": "accurate"}
        if patient:
            params["patient"] = patient
        url = f"{resource_type}?{urlencode(params)}"
        return self.search_url(url, resource_type)

    def search_url(self, url, resource_type):
        seen, resources = set(), {}
        expected_total = None
        while url:
            if url in seen:
                raise ValueError("FHIR pagination cycle")
            seen.add(url)
            bundle = self.request(url)
            if bundle.get("resourceType") != "Bundle":
                raise ValueError("Search did not return a Bundle")
            if bundle.get("total") is not None:
                total = bundle["total"]
                if not isinstance(total, int) or isinstance(total, bool) or total < 0:
                    raise ValueError("Invalid FHIR search total")
                if expected_total is not None and total != expected_total:
                    raise ValueError("FHIR search total changed during pagination")
                expected_total = total
            for entry in bundle.get("entry", []):
                if entry.get("search", {}).get("mode") == "outcome":
                    raise ValueError(f"FHIR search warning/error: {entry.get('resource')}")
                resource = entry.get("resource", {})
                if resource.get("resourceType") == "OperationOutcome":
                    raise ValueError(f"FHIR search warning/error: {resource}")
                if resource.get("resourceType") == resource_type:
                    resources[f"{resource_type}/{resource['id']}"] = resource
            next_url = next((link["url"] for link in bundle.get("link", []) if link.get("relation") == "next"), None)
            current_url = urljoin(self.base + "/", url)
            url = urljoin(current_url, next_url) if next_url else None
        if expected_total is not None and len(resources) != expected_total:
            raise ValueError(f"Incomplete FHIR {resource_type} search: received {len(resources)} of "
                             f"{expected_total} matches; pagination ended before all records were returned")
        return list(resources.values())


def clinical_content(resource):
    # Servers add narrative/version metadata; compare all remaining clinical fields.
    return {k: v for k, v in resource.items() if k not in {"meta", "text"}}


def check_capability(client):
    capability = client.request("metadata")
    if capability.get("resourceType") != "CapabilityStatement" or not capability.get("fhirVersion", "").startswith("4.0"):
        raise ValueError("Expected a FHIR R4 CapabilityStatement")
    return capability


def check_import_search(client, key, resource):
    """Fail on an unreadable/unsearchable imported record, with bounded diagnostics."""
    patient, kind = patient_ref(resource), resource["resourceType"]
    params = {"_id": resource["id"], "patient": patient, "_count": 2, "_total": "accurate"}
    found = client.search_url(kind + "?" + urlencode(params), kind)
    if len(found) == 1 and clinical_content(found[0]) == clinical_content(resource):
        print(f"Imported-record search probe passed: {key}", flush=True)
        return
    diagnostics = {"resource": key, "expected_patient": patient, "scoped_matches": len(found)}
    timeout = client.timeout
    try:
        client.timeout = min(timeout, 15)
        try:
            direct = client.request(key)
            diagnostics["read"] = {"patient": patient_ref(direct),
                "matches_source": clinical_content(direct) == clinical_content(resource)}
        except (OSError, ValueError, KeyError) as exc:
            diagnostics["read"] = {"error": str(exc)}
        for name, filters in (("by_id", {"_id": resource["id"]}),
                              ("bare_patient_id", {"_id": resource["id"], "patient": patient.split("/", 1)[1]})):
            try:
                bundle = client.request(kind + "?" + urlencode(dict(filters, _count=2, _total="accurate")))
                diagnostics[name] = {"total": bundle.get("total"), "resources": [
                    {"id": e.get("resource", {}).get("id"), "patient": patient_ref(e.get("resource", {}))}
                    for e in bundle.get("entry", [])[:2]]}
            except (OSError, ValueError, KeyError) as exc:
                diagnostics[name] = {"error": str(exc)}
    finally:
        client.timeout = timeout
    raise ValueError("Imported-record search probe failed: " + canonical(diagnostics))


def import_snapshot(snapshot, client, output, batch_size, *, probe_search=False):
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    check_capability(client)
    # Resume only when every existing patient belongs to this cohort.
    existing = {f"Patient/{r['id']}" for r in client.search_all("Patient")}
    foreign = existing - set(snapshot.patients)
    if foreign:
        raise ValueError(f"Server contains {len(foreign)} patients outside this snapshot. Use a dedicated server.")
    rows = iter(snapshot.resources.items())
    total = len(snapshot.resources)
    report = {"snapshot_id": snapshot.id, "base_url": client.base, "complete": False, "resources_imported": 0,
              "id_mapping": {key: key for key in snapshot.patients},
              "id_strategy": "PUT preserves original FHIR logical IDs; no server-ID remapping."}
    if hasattr(snapshot, "scope"):
        report["scope"] = snapshot.scope
    write_json(output, report)
    search_probed = False
    for start in range(0, total, batch_size):
        chunk = list(islice(rows, batch_size))
        bundle = {"resourceType": "Bundle", "type": "transaction", "entry": [
            {"fullUrl": f"{client.base}/{key}", "resource": resource,
             "request": {"method": "PUT", "url": key}} for key, resource in chunk
        ]}
        result = client.request(body=bundle)
        entries = result.get("entry", [])
        if result.get("type") != "transaction-response" or len(entries) != len(chunk):
            raise ValueError("Incomplete transaction response")
        if any(not e.get("response", {}).get("status", "").startswith(("200", "201")) for e in entries):
            raise ValueError("Import transaction contains unsuccessful responses")
        report["resources_imported"] = start + len(chunk)
        write_json(output, report)
        print(f"Imported {report['resources_imported']}/{total} resources", flush=True)
        if probe_search and not search_probed:
            sample = next(((key, resource) for key, resource in chunk
                           if resource.get("resourceType") in PATIENT_TYPES and patient_ref(resource)), None)
            if sample:
                check_import_search(client, *sample)
                search_probed = True
    report["complete"] = True
    write_json(output, report)
    return report


def verify_dataset(snapshot, rows, client):
    check_capability(client)
    patients = sorted({r["patient_id"] for r in rows})
    errors = []
    question_results = []
    checked = 0
    for patient in patients:
        print(f"Verifying {patient} ({checked + len(errors) + 1}/{len(patients)})", flush=True)
        try:
            # Retrieve all relevant resource types with pagination. Extra live records
            # invalidate absence/count labels just as missing records do.
            types = PATIENT_TYPES
            live_resources = [client.request(patient)]
            for resource_type in sorted(types):
                fetched = client.search_all(resource_type, patient)
                if any(patient_ref(r) != patient for r in fetched):
                    raise ValueError("Server ignored patient scope or returned wrong-patient resources")
                live_resources.extend(fetched)
            expected = {patient: clinical_content(snapshot.resources[patient])}
            for resource_type in types:
                expected.update({f"{r['resourceType']}/{r['id']}": clinical_content(r)
                                 for r in snapshot.for_patient(patient, resource_type)})
            actual = {f"{r['resourceType']}/{r['id']}": clinical_content(r) for r in live_resources}
            if expected != actual:
                missing, extra = set(expected) - set(actual), set(actual) - set(expected)
                changed = [key for key in set(expected) & set(actual) if expected[key] != actual[key]]
                from collections import Counter
                missing_types = dict(Counter(key.split('/')[0] for key in missing))
                raise ValueError(f"Snapshot differs: missing={len(missing)}, extra={len(extra)}, changed={len(changed)}; "
                                 f"missing_by_type={missing_types}; sample_missing={sorted(missing)[:5]}")
            # Resolve referenced resources needed by the answer rules, including
            # medication representations absent from patient-scoped searches.
            refs = {ref for row in rows if row["patient_id"] == patient
                    for ref in row["ground_truth_resource_ids"] if ref not in actual}
            refs.update(r.get("medicationReference", {}).get("reference") for r in live_resources
                        if r.get("resourceType") == "MedicationRequest")
            for ref in sorted(ref for ref in refs if ref and not ref.startswith("#")):
                resource = client.request(ref)
                if clinical_content(resource) != clinical_content(snapshot.resources[ref]):
                    raise ValueError(f"Referenced resource differs from snapshot: {ref}")
                if ref not in actual:
                    live_resources.append(resource)
            # Recompute every answer and evidence set using only HTTP-fetched bodies.
            live_snapshot = Snapshot(live_resources)
            patient_results = []
            for row in [r for r in rows if r["patient_id"] == patient]:
                answer, evidence = evaluate(live_snapshot, patient, row["oracle_spec"])
                if answer != row["ground_truth_answer"] or sorted(evidence) != sorted(row["ground_truth_resource_ids"]):
                    raise ValueError(f"Live answer/evidence mismatch for {row['id']}")
                patient_results.append({"id": row["id"], "family": row["family"],
                                        "answer": answer, "evidence_size": len(evidence),
                                        "answer_verified": True})
            question_results.extend(patient_results)
            checked += 1
        except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
            errors.append({"patient_id": patient, "error": str(exc)})
    return {"valid": not errors and bool(rows) and len(question_results) == len(rows),
            "snapshot_id": snapshot.id, "base_url": client.base,
            "patients_checked": checked, "patients_requested": len(patients),
            "questions_checked": len(question_results), "questions_requested": len(rows),
            "question_results": question_results, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["import", "verify"])
    add_source_arguments(parser)
    parser.add_argument("--base-url", required=True, help="Dedicated FHIR R4 server URL; no implicit production target")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    try:
        with open_records(args.bundles, args.snapshot) as snapshot:
            client = FHIR(args.base_url)
            if args.action == "import":
                if args.dataset:
                    from dataset.scope import scope_for_patients
                    snapshot = scope_for_patients(snapshot, {row["patient_id"] for row in
                                                  read_questions(args.dataset / "questions.jsonl")})
                import_snapshot(snapshot, client, args.output, args.batch_size)
            else:
                if not args.dataset:
                    raise ValueError("--dataset is required for verify")
                report = verify_dataset(snapshot, read_questions(args.dataset / "questions.jsonl"), client)
                write_json(args.output, report)
                if not report["valid"]:
                    parser.exit(1, f"Server verification failed; see {args.output}\n")
                print(f"Verified {report['patients_checked']} patients against snapshot")
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
