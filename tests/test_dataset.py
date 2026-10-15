"""Answer and integrity tests using explicit miniature FHIR fixtures, not Synthea data."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from dataset.common import Snapshot, canonical, read_questions, write_questions, iter_agent_inputs, digest, file_digest, jsonl_digests, sequence_digest, write_json, write_jsonl
from dataset.bundles import iter_bundle_entries
from dataset.cohort import generation_command
from dataset.fhir import FHIR, import_snapshot, verify_dataset
from dataset.oracle import InvalidExample, IneligibleExample, evaluate, quantity, chart_for
from dataset.questions import build_questions, save_dataset, split_for
from dataset.snapshot import build_snapshot, open_records
from dataset.validate import validate
from dataset.review import source_snapshot


def observation(rid, value, date, code="29463-7", unit="kg", patient="p1", **kwargs):
    return {
        "resourceType": "Observation", "id": rid, "status": "final",
        "subject": {"reference": f"Patient/{patient}"},
        "code": {"coding": [{"system": "http://loinc.org", "code": code}]},
        "effectiveDateTime": date,
        "valueQuantity": {"value": value, "unit": unit, "code": unit, "system": "http://unitsofmeasure.org"},
        **kwargs,
    }


def base_fixture():
    return [
        {"resourceType": "Patient", "id": "p1", "birthDate": "1980-01-01", "gender": "female",
         "identifier": [{"system": "urn:example:patient", "value": "p1"}]},
        {"resourceType": "Patient", "id": "p2", "birthDate": "1990-02-02", "gender": "male",
         "identifier": [{"system": "urn:example:patient", "value": "p2"}]},
        observation("old", 70, "2025-01-01T00:00:00Z"),
        observation("new", 75, "2025-12-01T00:00:00Z", encounter={"reference": "Encounter/e1"}),
        observation("bmi", 24.7, "2025-12-01T00:00:00Z", code="39156-5", unit="kg/m2"),
        observation("hr", 80, "2025-12-01T00:00:00Z", code="8867-4", unit="/min"),
        observation("a1c", 6.2, "2025-12-01T00:00:00Z", code="4548-4", unit="%"),
        {"resourceType": "Observation", "id": "bp", "status": "final",
         "subject": {"reference": "Patient/p1"}, "effectiveDateTime": "2025-12-01T00:00:00Z",
         "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9"}]},
         "component": [{"code": {"coding": [{"system": "http://loinc.org", "code": "8480-6"}]},
                        "valueQuantity": {"value": 120, "code": "mm[Hg]", "system": "http://unitsofmeasure.org"}}]},
        {"resourceType": "Encounter", "id": "e1", "status": "finished",
         "subject": {"reference": "Patient/p1"},
         "period": {"start": "2025-12-01T00:00:00Z", "end": "2025-12-02T00:00:00Z"},
         "serviceProvider": {"reference": "Organization/org"}},
        {"resourceType": "Organization", "id": "org", "name": "Fixture Hospital",
         "identifier": [{"system": "urn:example:org", "value": "org"}]},
        {"resourceType": "Condition", "id": "c1", "subject": {"reference": "Patient/p1"},
         "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]}},
        {"resourceType": "MedicationRequest", "id": "m1", "status": "completed", "subject": {"reference": "Patient/p1"}},
    ]


def fixture():
    resources = base_fixture()
    resources[8]["class"] = {"code": "AMB"}
    resources[7]["encounter"] = {"reference": "Encounter/e1"}
    resources[7]["component"].append({"code": {"coding": [{"system": "http://loinc.org", "code": "8462-4"}]},
                                      "valueQuantity": {"value": 80, "code": "mm[Hg]", "system": "http://unitsofmeasure.org"}})
    def encounter(rid, start, end, kind="IMP"):
        return {"resourceType": "Encounter", "id": rid, "status": "finished", "class": {"code": kind},
                "subject": {"reference": "Patient/p1"}, "period": {"start": start, "end": end}}
    def condition(rid, code, enc, onset):
        return {"resourceType": "Condition", "id": rid, "subject": {"reference": "Patient/p1"},
                "encounter": {"reference": "Encounter/" + enc}, "onsetDateTime": onset,
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": "Condition " + code}]}}
    def medication(code):
        return {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": code, "display": "Product " + code}]}
    def order(rid, enc, code, ref=False, reason=False):
        return {"resourceType": "MedicationRequest", "id": rid, "status": "completed", "intent": "order",
                "subject": {"reference": "Patient/p1"}, "encounter": {"reference": "Encounter/" + enc},
                "authoredOn": {"h0": "2025-01-11T00:00:00Z", "f1": "2025-02-15T12:00:00Z"}.get(enc, "2025-02-02T00:00:00Z"),
                **({"medicationReference": {"reference": "Medication/med-" + code}} if ref else {"medicationCodeableConcept": medication(code)}),
                **({"reasonReference": [{"reference": "Condition/diabetes"}]} if reason else {})}
    resources.extend([
        encounter("e0", "2024-12-20T09:00:00Z", "2024-12-20T11:00:00Z", "AMB"),
        encounter("h0", "2025-01-10T00:00:00Z", "2025-01-12T00:00:00Z"),
        encounter("h1", "2025-02-01T00:00:00Z", "2025-02-04T12:00:00Z"),
        encounter("f1", "2025-02-15T12:00:00Z", "2025-02-15T13:00:00Z", "AMB"),
        condition("diabetes", "44054006", "h0", "2024-12-15T00:00:00Z"),
        condition("diabetes-repeat", "44054006", "h1", "2025-02-01T00:00:00Z"),
        condition("followup-condition", "38341003", "f1", "2025-02-15T00:00:00Z"),
        {"resourceType": "Medication", "id": "med-A", "code": medication("A")},
        {"resourceType": "Medication", "id": "med-B", "code": medication("B")},
        order("h0-a", "h0", "A", ref=True, reason=True), order("h1-a", "h1", "A"),
        order("h1-b", "h1", "B", ref=True, reason=True), order("f1-b", "f1", "B"), order("f1-c", "f1", "C"),
        {"resourceType": "Procedure", "id": "proc", "status": "completed", "subject": {"reference": "Patient/p1"},
         "encounter": {"reference": "Encounter/h1"}, "performedDateTime": "2025-02-02T00:00:00Z",
         "reasonReference": [{"reference": "Condition/diabetes"}],
         "code": {"coding": [{"system": "http://snomed.info/sct", "code": "proc", "display": "Fixture procedure"}]}},
        observation("cr-baseline", 1.25, "2024-12-20T10:00:00Z", "2160-0", "mg/dL", encounter={"reference": "Encounter/e0"}),
        observation("cr-followup", 0.95, "2025-02-15T12:00:00Z", "2160-0", "mg/dL", encounter={"reference": "Encounter/f1"}),
        observation("glycemic-baseline", 7.2, "2024-12-10T00:00:00Z", "4548-4", "%"),
        observation("glycemic-followup", 6.8, "2025-01-20T00:00:00Z", "4548-4", "%"),
    ])
    for index, date, hb, platelets in [(0, "2025-01-11T00:00:00Z", 10.25, 180), (1, "2025-02-02T00:00:00Z", 11.35, 210)]:
        resources.extend([observation(f"hb{index}", hb, date, "718-7", "g/dL"),
                          observation(f"plt{index}", platelets, date, "777-3", "10*3/uL"),
                          {"resourceType": "DiagnosticReport", "id": f"cbc{index}", "status": "final",
                           "subject": {"reference": "Patient/p1"}, "effectiveDateTime": date,
                           "code": {"coding": [{"system": "http://loinc.org", "code": "58410-2"}]},
                           "result": [{"reference": f"Observation/hb{index}"}, {"reference": f"Observation/plt{index}"}]}])
    return resources


def config():
    return {"split_seed": "beyond-triage-v1", "split_ratios": [0.7, 0.15, 0.15], "end_date": "20261001"}


class ClinicalTests(unittest.TestCase):
    def answer(self, operation, resources=None):
        return evaluate(Snapshot(fixture() if resources is None else resources), "Patient/p1",
                        {"operation": operation, "as_of": "2026-10-02T00:00:00+00:00"})

    def test_all_fourteen_clinical_families_have_grounded_answers(self):
        rows, _, _, report = build_questions(Snapshot(fixture()), config())
        self.assertEqual(len({r["family"] for r in rows}), 14)
        self.assertEqual(len(rows), 14)
        self.assertEqual(set(report["complexity"]), {"simple", "intermediate", "complex"})
        self.assertTrue(all(isinstance(r["ground_truth_answer"], dict) for r in rows))

    def test_reference_and_inline_medications_resolve_to_same_product(self):
        answer, evidence = self.answer("admission_medication_comparison")
        self.assertEqual([r["code"] for r in answer["products"]["both"]], ["A"])
        self.assertEqual([r["code"] for r in answer["products"]["only_second"]], ["B"])
        self.assertIn("Medication/med-A", evidence)
        self.assertIn("Medication/med-B", evidence)

    def test_report_results_are_not_replaced_by_newer_unlinked_observations(self):
        resources = fixture() + [observation("unlinked", 999, "2026-01-01", "718-7", "g/dL")]
        answer, evidence = self.answer("cbc_panel", resources)
        self.assertEqual(answer["hemoglobin"]["value"], "11.35")
        self.assertNotIn("Observation/unlinked", evidence)
        self.assertIn("DiagnosticReport/cbc1", evidence)

    def test_component_extraction_is_scoped_to_the_outpatient_encounter(self):
        answer, evidence = self.answer("outpatient_bp")
        self.assertEqual(answer["systolic"]["value"], "120")
        self.assertEqual(answer["diastolic"]["value"], "80")
        self.assertIn("Encounter/e1", evidence)

    def test_followup_compares_encounter_linked_orders(self):
        answer, evidence = self.answer("followup_medication_comparison")
        self.assertEqual(answer["followup"]["days_after_discharge"], "11")
        self.assertEqual([r["code"] for r in answer["products"]["both"]], ["B"])
        self.assertEqual([r["code"] for r in answer["products"]["only_first"]], ["A"])
        self.assertEqual([r["code"] for r in answer["products"]["only_second"]], ["C"])

    def test_no_followup_retains_evidence_for_medication_bearing_anchor(self):
        resources = [r for r in fixture() if r["id"] != "f1"]
        answer, evidence = self.answer("followup_medication_comparison", resources)
        self.assertIsNone(answer["followup"])
        self.assertNotIn("products", answer)
        self.assertIn("Encounter/h1", evidence)
        self.assertIn("MedicationRequest/h1-a", evidence)
        self.assertIn("MedicationRequest/h1-b", evidence)
        self.assertNotIn("MedicationRequest/f1-b", evidence)

    def test_medication_followup_skips_newer_stays_without_orders(self):
        resources = fixture()
        later = copy.deepcopy(next(r for r in resources if r["id"] == "h1"))
        later.update(id="empty-stay", period={"start": "2025-06-01", "end": "2025-06-02"})
        answer, _ = self.answer("followup_medication_comparison", resources + [later])
        self.assertEqual(answer["admitted_at"], "2025-02-01T00:00:00+00:00")
        self.assertTrue(answer["products"]["both"] or answer["products"]["only_first"])
        resources = [r for r in resources if not (r.get("resourceType") == "MedicationRequest"
                     and r.get("encounter", {}).get("reference") in {"Encounter/h0", "Encounter/h1"})]
        with self.assertRaises(IneligibleExample):
            self.answer("followup_medication_comparison", resources)

    def test_procedure_indications_require_explicit_condition_links(self):
        resources = fixture()
        unrelated = copy.deepcopy(next(r for r in resources if r["id"] == "proc"))
        unrelated["id"] = "unlinked-procedure"
        unrelated.pop("reasonReference")
        answer, evidence = self.answer("procedure_indications", resources + [unrelated])
        self.assertEqual(len(answer["procedures"]), 1)
        self.assertEqual(answer["procedures"][0]["indications"][0]["code"], "44054006")
        self.assertIn("Condition/diabetes", evidence)
        self.assertNotIn("Procedure/unlinked-procedure", evidence)
        self.assertFalse(any(ref.startswith("MedicationRequest/") for ref in evidence))
        next(r for r in resources if r["id"] == "proc")["reasonReference"] = [{"reference": "Observation/hb1"}]
        with self.assertRaises(IneligibleExample):
            self.answer("procedure_indications", resources)

    def test_preadmission_treatment_requires_recent_explicit_prescribing_evidence(self):
        resources = fixture()
        social = copy.deepcopy(next(r for r in resources if r["id"] == "diabetes"))
        social.update(id="social-finding", code={"coding": [{"system": "http://snomed.info/sct",
                                                            "code": "160903007", "display": "Full-time employment"}]})
        answer, evidence = self.answer("preadmission_treated_conditions", resources + [social])
        selected = answer["conditions_with_prior_orders"]
        self.assertEqual([r["condition"]["code"] for r in selected], ["44054006"])
        self.assertEqual([r["medication"]["code"] for r in selected[0]["orders"]], ["A"])
        self.assertIn("Medication/med-A", evidence)
        self.assertNotIn("Condition/social-finding", evidence)
        self.assertNotIn("MedicationRequest/h1-b", evidence)
        next(r for r in resources if r["id"] == "diabetes")["abatementDateTime"] = "2025-01-31"
        answer, _ = self.answer("preadmission_treated_conditions", resources)
        self.assertEqual(answer["conditions_with_prior_orders"], [])

    def test_preadmission_prescribing_window_boundaries_and_invalid_orders(self):
        from datetime import timedelta
        from dataset.common import timestamp
        resources = fixture()
        admission = timestamp("2025-02-01")
        order = next(r for r in resources if r["id"] == "h0-a")
        next(r for r in resources if r["id"] == "diabetes")["onsetDateTime"] = "2020-01-01"
        order["authoredOn"] = (admission - timedelta(days=365)).isoformat()
        answer, _ = self.answer("preadmission_treated_conditions", resources)
        self.assertEqual(len(answer["conditions_with_prior_orders"]), 1)
        for authored, status in [((admission - timedelta(days=365, microseconds=1)).isoformat(), "completed"),
                                 (admission.isoformat(), "completed"), ("2025-01-11", "draft")]:
            order.update(authoredOn=authored, status=status)
            answer, _ = self.answer("preadmission_treated_conditions", resources)
            self.assertEqual(answer["conditions_with_prior_orders"], [])

    def test_cbc_trend_uses_latest_close_pair_instead_of_distant_latest_report(self):
        resources = fixture()
        distant = copy.deepcopy(next(r for r in resources if r["id"] == "cbc1"))
        distant.update(id="distant-report", effectiveDateTime="2026-01-01")
        answer, evidence = self.answer("cbc_change", resources + [distant])
        self.assertEqual(answer["latest"]["report_time"], "2025-02-02T00:00:00+00:00")
        self.assertNotIn("DiagnosticReport/distant-report", evidence)
        next(r for r in resources if r["id"] == "cbc0")["effectiveDateTime"] = "2024-01-01"
        with self.assertRaises(IneligibleExample):
            self.answer("cbc_change", resources)

    def test_cbc_trend_includes_ninety_day_boundary(self):
        from datetime import timedelta
        from dataset.common import timestamp
        resources = fixture()
        earlier = next(r for r in resources if r["id"] == "cbc0")
        boundary = timestamp("2025-02-02") - timedelta(days=90)
        earlier["effectiveDateTime"] = boundary.isoformat()
        answer, _ = self.answer("cbc_change", resources)
        self.assertEqual(answer["previous"]["report_time"], boundary.isoformat())
        earlier["effectiveDateTime"] = (boundary - timedelta(microseconds=1)).isoformat()
        with self.assertRaises(IneligibleExample):
            self.answer("cbc_change", resources)

    def test_missing_value_is_distinct_from_missing_observation(self):
        resources = fixture()
        next(r for r in resources if r["id"] == "cr-followup").pop("valueQuantity")
        answer, evidence = self.answer("renal_followup", resources)
        self.assertEqual(answer["followup"]["status"], "missing_value")
        self.assertIsNone(answer["change_mg_dL"])
        self.assertIn("Observation/cr-followup", evidence)
        resources = [r for r in resources if r["id"] != "cr-followup"]
        answer, _ = self.answer("renal_followup", resources)
        self.assertEqual(answer["followup"]["status"], "no_matching_record")
        self.assertIn("baseline", answer)

    def test_creatinine_and_report_changes_use_unrounded_values(self):
        answer, _ = self.answer("renal_followup")
        self.assertEqual(answer["change_mg_dL"], "-0.3")
        answer, _ = self.answer("cbc_change")
        self.assertEqual(answer["hemoglobin_change_g_dL"], "1.1")
        self.assertEqual(answer["platelet_change_10_3_uL"], "30")

    def test_subsequent_admission_uses_earliest_stay_and_links_conditions(self):
        answer, evidence = self.answer("subsequent_admission")
        self.assertEqual(answer["subsequent_admission"]["admitted_at"], "2025-02-01T00:00:00+00:00")
        self.assertEqual(answer["subsequent_admission"]["shared_conditions"][0]["code"], "44054006")
        self.assertIn("Condition/diabetes-repeat", evidence)

    def test_glycemic_baseline_and_followup(self):
        answer, _ = self.answer("glycemic_followup")
        self.assertEqual(answer["baseline"]["value"], "7.2")
        self.assertEqual(answer["followup"]["value"], "6.8")
        self.assertEqual(answer["change_percentage_points"], "-0.4")

    def test_chart_cache_does_not_retain_completed_patient_snapshots(self):
        import weakref
        snapshot = Snapshot(fixture())
        reference = weakref.ref(snapshot)
        evaluate(snapshot, "Patient/p1", {"operation": "cbc_panel", "as_of": "2026-10-02"})
        del snapshot
        self.assertIsNone(reference())

    def test_monitoring_window_excludes_upper_boundary_and_requires_complete_window(self):
        from datetime import timedelta
        from dataset.common import timestamp
        resources = fixture()
        anchor = timestamp("2024-12-20T10:00:00Z")
        # A measurement exactly on day 30 is excluded for this lab-window task.
        following = next(r for r in resources if r["id"] == "cr-followup")
        following["effectiveDateTime"] = (anchor + timedelta(days=30)).isoformat()
        spec = {"operation": "renal_followup", "followup_days": 30, "as_of": "2026-10-02"}
        answer, _ = evaluate(Snapshot(resources), "Patient/p1", spec)
        self.assertEqual(answer["followup"]["status"], "no_matching_record")
        spec["as_of"] = "2025-01-01"
        with self.assertRaises(IneligibleExample):
            evaluate(Snapshot(resources), "Patient/p1", spec)

    def test_discharge_followup_includes_last_instant_and_excludes_discharge_itself(self):
        from datetime import timedelta
        from dataset.common import timestamp
        resources = fixture()
        discharge = timestamp("2025-02-04T12:00:00Z")
        followup = next(r for r in resources if r["id"] == "f1")
        followup["period"]["start"] = (discharge + timedelta(days=30)).isoformat()
        followup["period"]["end"] = (discharge + timedelta(days=30, hours=1)).isoformat()
        answer, _ = self.answer("discharge_followup", resources)
        self.assertEqual(answer["followup"]["days_after_discharge"], "30")
        followup["period"]["start"] = discharge.isoformat()
        answer, _ = self.answer("discharge_followup", resources)
        self.assertIsNone(answer["followup"])

    def test_measurement_time_and_reference_match_are_both_required(self):
        resources = fixture()
        # A value outside the requested temporal interval cannot stand in for a missing one.
        next(r for r in resources if r["id"] == "cr-followup")["effectiveDateTime"] = "2027-01-01"
        answer, _ = self.answer("renal_followup", resources)
        self.assertEqual(answer["followup"]["status"], "no_matching_record")

    def test_prescribing_indications_require_explicit_reference(self):
        answer, evidence = self.answer("medication_indications")
        self.assertEqual(len(answer["orders"]), 1)
        self.assertEqual(answer["orders"][0]["indications"][0]["code"], "44054006")
        self.assertNotIn("MedicationRequest/h1-a", evidence)

    def test_wrong_patient_and_broken_reference_are_rejected(self):
        resources = fixture()
        next(r for r in resources if r["id"] == "hb1")["subject"]["reference"] = "Patient/p2"
        with self.assertRaisesRegex(InvalidExample, "patient boundary"):
            self.answer("cbc_panel", resources)
        resources = [r for r in fixture() if r["id"] != "med-B"]
        with self.assertRaisesRegex(InvalidExample, "Unresolved"):
            self.answer("inpatient_medications", resources)

    def test_ambiguous_anchors_and_conflicting_measurements_are_rejected(self):
        resources = fixture()
        duplicate = copy.deepcopy(next(r for r in resources if r["id"] == "cbc1"))
        duplicate["id"] = "tied-report"
        with self.assertRaisesRegex(InvalidExample, "Ambiguous"):
            self.answer("cbc_panel", resources + [duplicate])
        resources.append(observation("conflict", 150, "2025-12-01T00:00:00Z", "8480-6", "mm[Hg]", encounter={"reference": "Encounter/e1"}))
        with self.assertRaisesRegex(InvalidExample, "Conflicting"):
            self.answer("outpatient_bp", resources)


class PipelineTests(unittest.TestCase):
    def write_bundles(self, directory, resources):
        directory.mkdir()
        entries = [{"fullUrl": f"urn:uuid:{r['id']}", "resource": r} for r in resources]
        write_json(directory / "patients.json", {"resourceType": "Bundle", "type": "transaction", "entry": entries})

    def test_streamed_hash_matches_canonical_array_hash(self):
        self.assertEqual(sequence_digest(iter(fixture())), digest(fixture()))
        self.assertEqual(sequence_digest(iter([])), digest([]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resources.jsonl"
            write_jsonl(path, fixture())
            self.assertEqual(jsonl_digests(path), (file_digest(path), digest(fixture())))

    def test_bundle_streaming_handles_large_entries_and_field_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.json"
            entries = [{"resource": {"resourceType": "DocumentReference", "id": "large",
                                     "text": {"div": "x" * 200000}}},
                       {"resource": {"resourceType": "Patient", "id": "p"}}]
            write_json(path, {"resourceType": "Bundle", "entry": entries})
            self.assertEqual(list(iter_bundle_entries(path)), entries)
            path.write_text('{"resourceType":"Bundle","entry":[{},]}')
            with self.assertRaises(ValueError):
                list(iter_bundle_entries(path))
            path.write_text('{"resourceType":"Patient","entry":[]}')
            with self.assertRaises(ValueError):
                list(iter_bundle_entries(path))

    def test_snapshot_normalizes_references_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = fixture()
            for r in resources:
                if r.get("subject"):
                    r["subject"]["reference"] = "urn:uuid:p1"
                if r["id"] == "e1":
                    r["serviceProvider"]["reference"] = "Organization?identifier=urn%3Aexample%3Aorg%7Corg"
            self.write_bundles(root / "bundles", resources)
            build_snapshot([root / "bundles"], root / "snapshot")
            snapshot = Snapshot.load(root / "snapshot")
            self.assertEqual(snapshot.resources["Encounter/e1"]["serviceProvider"]["reference"], "Organization/org")
            self.assertEqual(snapshot.resources["Observation/new"]["subject"]["reference"], "Patient/p1")
            with (root / "snapshot/resources.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaises(ValueError):
                Snapshot.load(root / "snapshot")

    def test_temporary_source_index_matches_snapshot_and_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_bundles(root / "bundles", fixture())
            original = build_snapshot([root / "bundles"], root / "explicit-snapshot")
            with open_records([root / "bundles"]) as records:
                databases = records.resources.connection.execute("PRAGMA database_list").fetchall()
                index = Path(databases[0][2])
                self.assertTrue(index.exists())
                self.assertEqual(records.id, original["snapshot_id"])
                self.assertEqual(set(index.parent.iterdir()), {index})
                rows, _, splits, _ = build_questions(records, config())
                self.assertTrue(rows)
            self.assertFalse(index.exists())
            with self.assertRaisesRegex(ValueError, "simulated"):
                with open_records([root / "bundles"]) as records:
                    index = Path(records.resources.connection.execute("PRAGMA database_list").fetchone()[2])
                    raise ValueError("simulated")
            self.assertFalse(index.parent.exists())

    def test_snapshot_rejects_incomplete_and_duplicate_patient_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = fixture()
            resources = [r for r in resources if r["resourceType"] != "Organization"]
            self.write_bundles(root / "bundles", resources)
            with self.assertRaisesRegex(ValueError, "unresolved"):
                build_snapshot([root / "bundles"], root / "snapshot")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = fixture()
            resources[1]["identifier"] = resources[0]["identifier"]
            self.write_bundles(root / "bundles", resources)
            with self.assertRaisesRegex(ValueError, "Duplicate patient"):
                build_snapshot([root / "bundles"], root / "snapshot")

    def test_missing_preserved_bundle_is_not_accepted_as_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_bundles(root / "fhir", fixture())
            write_json(root / "generation.json", {"status": "complete", "files": {
                "fhir/patients.json": "wrong", "fhir/missing.json": "missing"}})
            with self.assertRaisesRegex(ValueError, "Missing/extra"):
                build_snapshot([root / "fhir"], root / "snapshot")

    def test_two_file_dataset_and_safe_agent_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_bundles(root / "bundles", fixture())
            build_snapshot([root / "bundles"], root / "snapshot")
            snapshot = Snapshot.load(root / "snapshot")
            rows, rejected, splits, report = build_questions(snapshot, config())
            save_dataset(root / "dataset", rows, splits)
            self.assertTrue(validate(snapshot, root / "dataset")["valid"])
            self.assertEqual({p.name for p in (root / "dataset").iterdir()},
                             {"questions.jsonl", "patient_splits.json"})
            splits_path = root / "dataset/patient_splits.json"
            self.assertEqual(json.loads(splits_path.read_text()), splits)
            self.assertEqual(read_questions(root / "dataset/questions.jsonl"), rows)
            expected = [{key: row[key] for key in ("id", "question", "patient_id", "split")}
                        for row in rows]
            self.assertEqual(list(iter_agent_inputs(root / "dataset")), expected)
            for split in ("train", "validation", "test"):
                self.assertEqual(list(iter_agent_inputs(root / "dataset", split)),
                                 [row for row in expected if row["split"] == split])
            self.assertEqual(json.loads(splits_path.read_text()), splits)

    def test_replace_updates_current_files_and_refuses_unrelated_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "current"
            rows, _, splits, _ = build_questions(Snapshot(fixture()), config())
            save_dataset(output, rows, splits)
            with self.assertRaisesRegex(ValueError, "exists"):
                save_dataset(output, rows, splits)
            save_dataset(output, rows, splits, replace=True)
            self.assertEqual({p.name for p in output.iterdir()}, {"questions.jsonl", "patient_splits.json"})
            self.assertEqual(read_questions(output / "questions.jsonl"), rows)
            (output / "notes.txt").write_text("unrelated work")
            with self.assertRaisesRegex(ValueError, "unrelated"):
                save_dataset(output, rows, splits, replace=True)
            self.assertEqual((output / "notes.txt").read_text(), "unrelated work")

    def test_question_jsonl_preserves_structures_and_large_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "questions.jsonl"
            rows, _, _, _ = build_questions(Snapshot(fixture()), config())
            row = copy.deepcopy(rows[0])
            row["question"] = 'Patient says "hello, café"\nWhich record?'
            row["ground_truth_resource_ids"] = [f"Observation/{i:036d}" for i in range(4000)]
            write_questions(path, [row])
            self.assertEqual(read_questions(path), [row])
            path.write_text('{"id":"bad"}\n')
            with self.assertRaisesRegex(ValueError, "schema"):
                read_questions(path)

    def test_validation_detects_tampered_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = Snapshot(fixture())
            cfg = config()
            rows, rejected, splits, report = build_questions(snapshot, cfg)
            save_dataset(root / "dataset", rows, splits)
            self.assertTrue(validate(snapshot, root / "dataset")["valid"])
            rows[0]["ground_truth_answer"] = {"wrong": True}
            write_questions(root / "dataset/questions.jsonl", rows)
            result = validate(snapshot, root / "dataset")
            self.assertTrue(any("Answer mismatch" in error for error in result["errors"]))

    def test_raw_source_replay_resolves_original_urn_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resources = fixture()
            for r in resources:
                if r.get("subject"):
                    r["subject"]["reference"] = "urn:uuid:p1"
            self.write_bundles(root / "bundles", resources)
            raw = source_snapshot(root / "bundles/patients.json")
            answer, evidence = evaluate(raw, "Patient/p1", {"operation": "cbc_panel", "as_of": "2026-10-02"})
            self.assertEqual(answer["hemoglobin"]["value"], "11.35")

    def test_split_assignments_and_question_ids_survive_expansion(self):
        rows, _, splits, _ = build_questions(Snapshot(fixture()), config(), 1)
        expanded, _, expanded_splits, _ = build_questions(Snapshot(fixture()), config())
        self.assertTrue(all(expanded_splits[p] == split for p, split in splits.items()))
        self.assertTrue({r["id"] for r in rows} <= {r["id"] for r in expanded})
        self.assertTrue(all(r["split"] == splits[r["patient_id"]] for r in rows))
        self.assertEqual(split_for("Patient/p1", **{"seed": config()["split_seed"], "ratios": config()["split_ratios"]}),
                         expanded_splits["Patient/p1"])

    def test_generation_command_pins_both_dates_and_disables_overflow(self):
        cfg = {"batch_seeds": [1, 2], "clinician_seed": 3, "patients_per_batch": 200,
               "reference_date": "20261001", "end_date": "20261001", "state": "Massachusetts"}
        cmd = generation_command(cfg, 2, Path("synthea.jar"), Path("output"), Path("config"))
        for option, expected in [("-s", "2"), ("-cs", "3"), ("-p", "200"),
                                 ("-r", "20261001"), ("-e", "20261001"), ("-o", "false")]:
            self.assertEqual(cmd[cmd.index(option) + 1], expected)

    def test_http_failure_never_becomes_zero_results(self):
        with patch("dataset.fhir.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(URLError):
                FHIR("http://127.0.0.1:8080/fhir").search_all("Observation", "Patient/p1")

    def test_pagination_and_outcome_errors(self):
        client = FHIR("http://127.0.0.1:8080/fhir")
        pages = [
            {"resourceType": "Bundle", "entry": [{"resource": fixture()[2]}],
             "link": [{"relation": "next", "url": "http://127.0.0.1:8080/fhir/Observation?page=2"}]},
            {"resourceType": "Bundle", "entry": [{"resource": fixture()[3]}]},
        ]
        with patch.object(client, "request", side_effect=pages):
            self.assertEqual(len(client.search_all("Observation", "Patient/p1")), 2)
        with patch.object(client, "request", return_value={
            "resourceType": "Bundle", "entry": [{"search": {"mode": "outcome"},
                                                "resource": {"resourceType": "OperationOutcome"}}]}):
            with self.assertRaises(ValueError):
                client.search_all("Observation")

    def test_relative_pagination_link_keeps_resource_path(self):
        client = FHIR("http://127.0.0.1:8080/fhir")
        pages = [
            {"resourceType": "Bundle", "entry": [], "link": [{"relation": "next", "url": "?page=2"}]},
            {"resourceType": "Bundle", "entry": []},
        ]
        with patch.object(client, "request", side_effect=pages) as request:
            client.search_all("Observation", "Patient/p1")
            self.assertEqual(request.call_args_list[1].args[0], "http://127.0.0.1:8080/fhir/Observation?page=2")

    def test_live_verification_detects_missing_and_extra_records(self):
        resources = fixture()
        snapshot = Snapshot(resources)
        rows, _, _, _ = build_questions(snapshot, config())
        rows = [r for r in rows if r["patient_id"] == "Patient/p1"]
        client = FHIR("http://127.0.0.1:8080/fhir")

        def request(path="", body=None):
            if path == "metadata":
                return {"resourceType": "CapabilityStatement", "fhirVersion": "4.0.1"}
            if "?" not in path:
                return snapshot.resources[path]
            raise AssertionError(f"Unexpected request: {path}")

        def search_all(rtype, patient=None):
            return [copy.deepcopy(r) for r in snapshot.for_patient(patient, rtype)]

        with patch.object(client, "request", side_effect=request), patch.object(client, "search_all", side_effect=search_all):
            self.assertTrue(verify_dataset(snapshot, rows, client)["valid"])

        def extra(rtype, patient=None):
            found = search_all(rtype, patient)
            if rtype == "Observation":
                found.append(observation("extra", 81, "2026-01-01"))
            return found

        with patch.object(client, "request", side_effect=request), patch.object(client, "search_all", side_effect=extra):
            self.assertFalse(verify_dataset(snapshot, rows, client)["valid"])

        def missing(rtype, patient=None):
            return [r for r in search_all(rtype, patient) if r["id"] != "new"]

        with patch.object(client, "request", side_effect=request), patch.object(client, "search_all", side_effect=missing):
            self.assertFalse(verify_dataset(snapshot, rows, client)["valid"])

    def test_import_preserves_ids_and_refuses_unrelated_server_patients(self):
        snapshot = Snapshot(fixture())
        client = FHIR("http://127.0.0.1:8080/fhir")
        transactions = []

        def request(path="", body=None):
            if path == "metadata":
                return {"resourceType": "CapabilityStatement", "fhirVersion": "4.0.1"}
            transactions.append(body)
            return {"resourceType": "Bundle", "type": "transaction-response",
                    "entry": [{"response": {"status": "201 Created"}} for _ in body["entry"]]}

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(client, "request", side_effect=request), patch.object(client, "search_all", return_value=[]):
                report = import_snapshot(snapshot, client, Path(tmp) / "report.json", 100)
            self.assertTrue(report["complete"])
            for entry in transactions[0]["entry"]:
                self.assertEqual(entry["request"]["method"], "PUT")
                self.assertEqual(entry["request"]["url"], f"{entry['resource']['resourceType']}/{entry['resource']['id']}")
            with patch.object(client, "request", side_effect=request), patch.object(
                    client, "search_all", return_value=[{"id": "unrelated"}]):
                with self.assertRaisesRegex(ValueError, "outside this snapshot"):
                    import_snapshot(snapshot, client, Path(tmp) / "report.json", 100)


if __name__ == "__main__":
    unittest.main()
