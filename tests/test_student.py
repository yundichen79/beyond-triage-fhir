"""Tool completeness, label isolation, answer semantics, and actual loop execution."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

from dataset.common import Snapshot, canonical, write_questions, file_digest
from dataset.fhir import FHIR
from dataset.live import run_job
from dataset.oracle import evaluate
from dataset.questions import build_questions, negative
from student.agent import ContextLimit, Generation, Settings, SYSTEM_PROMPT, parse_calls, run_agent
from student.run import execute_run, select_questions, selection_summary, summarize
from student.scoring import equal, score_episode
from student.tools import FIELDS, MAX_COUNT, PatientTools, TOOL_DEFINITIONS, projection, tool_definitions
from student.contract import answer_error, INTERFACE_VERSION, contract_for_question, json_schema
from test_dataset import fixture, config, observation
from fhir_fixture import search as fixture_search


CBC_QUESTION = "Review the latest final, amended, or corrected CBC report (LOINC 58410-2)."


class ScriptedModel:
    metadata = {"model": "scripted-test-only", "revision": None}

    def __init__(self, responses):
        self.responses, self.inputs, self.seeds, self.budgets = iter(responses), [], [], []

    def seed(self, value):
        self.seeds.append(value)

    def generate(self, messages, tools, *, max_new_tokens=None):
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        self.budgets.append(max_new_tokens)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return Generation(response, 100, min(20, max_new_tokens or 20))


def call(name, **arguments):
    return '<tool_call>\n' + canonical({"name": name, "arguments": arguments}) + '\n</tool_call>'


def fake_client(resources=None):
    snapshot = Snapshot(fixture() if resources is None else resources)
    client = FHIR("http://localhost:8080/fhir")
    def request(path):
        if "?" not in path:
            return copy.deepcopy(snapshot.resources[path])
        return fixture_search(snapshot, path)
    client.request = MagicMock(side_effect=request)
    return client


class ToolTests(unittest.TestCase):
    def test_calculator_reproduces_observed_cbc_rounding_and_half_even_ties(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        for earlier, later, expected in [("15.543", "11.259", "-4.3"),
                                          ("246.23", "444.78", "198.6"),
                                          ("0", "1.25", "1.2"),
                                          ("0", "-1.25", "-1.2")]:
            result = tools.execute("calculate", {"earlier": earlier, "later": later, "decimal_places": 1})
            self.assertEqual(result["value"], expected)
        client.request.assert_not_called()
        self.assertFalse(tools.seen)

    def test_calculator_rejects_invalid_inputs_without_http(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        for override in ({"earlier": "NaN"}, {"later": "1e999"}, {"later": True},
                         {"decimal_places": True}, {"decimal_places": 7}, {"extra": "x"}):
            result = tools.execute("calculate", {"earlier": "1", "later": "2", "decimal_places": 1} | override)
            self.assertIn("error", result)
        client.request.assert_not_called()

    def test_contradictory_single_valued_filters_fail_before_http(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        tools.allowed.update({"Encounter/e0", "Encounter/f1"})
        for kind, params in (("Encounter", {"class": ["IMP", "AMB"]}),
                             ("Observation", {"encounter": ["Encounter/e0", "Encounter/f1"]})):
            result = tools.execute("search_fhir", {"resource_type": kind, "params": params})
            self.assertIn("Contradictory", result["error"])
            self.assertIn("comma-separated", result["error"])
        client.request.assert_not_called()
        result = tools.execute("search_fhir", {"resource_type": "Observation", "params": {
            "encounter": "Encounter/e0,Encounter/f1", "code": "2160-0"}})
        self.assertNotIn("error", result)

    def test_invalid_resource_statuses_sorts_and_unseen_encounters_never_reach_http(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        cases = [
            ("Encounter", {"status": ["final", "amended", "corrected"]}, "Invalid Encounter status"),
            ("Procedure", {"status": "final,amended,corrected"}, "Invalid Procedure status"),
            ("MedicationRequest", {"status": "final"}, "Invalid MedicationRequest status"),
            ("Condition", {"clinical-status": "final"}, "Invalid Condition clinical-status"),
            ("Condition", {"verification-status": "final"}, "Invalid Condition verification-status"),
            ("Condition", {"_sort": "-date"}, "onset-date"),
            ("Observation", {"encounter": "Encounter/123456789"}, "Unseen encounter"),
            ("Observation", {"status": ["final", "amended"]}, "Contradictory status"),
        ]
        for kind, params, hint in cases:
            with self.subTest(kind=kind, params=params):
                result = tools.execute("search_fhir", {"resource_type": kind, "params": params})
                self.assertIn(hint, result["error"])
                self.assertFalse(result["complete"])
                self.assertNotIn("resources", result)
        client.request.assert_not_called()

    def test_filter_sensitive_transport_handles_codes_dates_classes_and_statuses(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        result = tools.execute("search_fhir", {"resource_type": "Observation", "params": {
            "code": "http://loinc.org|2160-0", "status": "final,amended,corrected",
            "date": ["ge2025-01-01", "lt2026-01-01"], "_sort": "date"}})
        self.assertEqual([r["id"] for r in result["resources"]], ["cr-followup"])
        encounters = tools.execute("search_fhir", {"resource_type": "Encounter", "params": {
            "class": "IMP", "status": "finished", "_sort": "-date"}})
        self.assertEqual([r["id"] for r in encounters["resources"]], ["h1", "h0"])
        # A genuinely valid but nonmatching query remains an empty successful result.
        empty = tools.execute("search_fhir", {"resource_type": "Procedure", "params": {"status": "in-progress"}})
        self.assertTrue(empty["complete"])
        self.assertEqual(empty["resources"], [])
        with self.assertRaisesRegex(AssertionError, "does not implement"):
            client.request("Observation?patient=Patient/p1&ignored-parameter=value")

    def test_encounter_scope_requires_discovery_and_preserves_or_reference_filters(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        result = tools.execute("search_fhir", {"resource_type": "Observation", "params": {"encounter": "Encounter/f1"}})
        self.assertIn("Unseen encounter", result["error"])
        tools.execute("search_fhir", {"resource_type": "Encounter", "params": {"_id": "e0,f1"}})
        result = tools.execute("search_fhir", {"resource_type": "Observation", "params": {
            "encounter": "Encounter/e0,Encounter/f1", "code": "2160-0", "_sort": "date"}})
        self.assertEqual([r["id"] for r in result["resources"]], ["cr-baseline", "cr-followup"])
        tools = PatientTools(client, "Patient/p1")
        tools.execute("search_fhir", {"resource_type": "MedicationRequest", "params": {"_id": "h1-b"}})
        self.assertIn("Encounter/h1", tools.allowed)
        self.assertNotIn("error", tools.execute("search_fhir", {
            "resource_type": "Condition", "params": {"encounter": "Encounter/h1"}}))

    def test_search_feedback_rejects_observed_bad_queries_before_http(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        cases = [
            ("MedicationRequest", {"reasonReference": "Condition/c1"}, "reasonReference"),
            ("Condition", {"date": "lt2026-01-01"}, "onset-date"),
            ("Encounter", {"sort": "date"}, "_sort"),
            ("Observation", {"code": {"system": "loinc", "code": "1"}}, "system|code"),
            ("Encounter", {"_sort": "date:asc"}, "-date"),
            ("Observation", {"encounter": "ambulatory"}, "Encounter/logical-id"),
            ("Procedure", {"date": ["lt", "2026-10-02"]}, "one date per string"),
            ("Encounter", {"date": "[ge2025-01-01,lt2026-01-01]"}, "one date per string"),
        ]
        for kind, params, hint in cases:
            with self.subTest(params=params):
                result = tools.execute("search_fhir", {"resource_type": kind, "params": params})
                self.assertFalse(result["complete"])
                self.assertIn(hint, result["error"])
        client.request.assert_not_called()
        self.assertFalse(tools.seen)

    def test_or_status_and_and_date_filters_are_not_rewritten(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        tools.execute("search_fhir", {"resource_type": "Encounter", "params": {"_id": "e1"}})
        result = tools.execute("search_fhir", {"resource_type": "Observation", "params": {
            "status": "final,amended,corrected", "date": ["ge2025-01-01", "lt2026-01-01"],
            "encounter": "Encounter/e1", "_sort": "-date"}})
        self.assertNotIn("error", result)
        query = parse_qs(urlparse(client.request.call_args.args[0]).query)
        self.assertEqual(query["status"], ["final,amended,corrected"])
        self.assertEqual(query["date"], ["ge2025-01-01", "lt2026-01-01"])
        self.assertEqual(query["patient"], ["Patient/p1"])

    def test_patient_scope_and_repeated_date_parameters_are_preserved(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        result = tools.execute("search_fhir", {"resource_type": "Observation",
            "params": {"date": ["ge2025-01-01", "lt2026-01-01"], "component-code": "8480-6"}, "count": 5})
        self.assertTrue(result["complete"])
        query = parse_qs(urlparse(client.request.call_args.args[0]).query)
        self.assertEqual(query["date"], ["ge2025-01-01", "lt2026-01-01"])
        self.assertEqual(query["patient"], ["Patient/p1"])
        self.assertEqual(query["_count"], ["5"])
        self.assertEqual(query["component-code"], ["8480-6"])
        for params in ({"patient": "Patient/p2"}, {"_include": "*"}):
            self.assertIn("error", tools.execute("search_fhir", {"resource_type": "Observation", "params": params}))

    def test_pagination_is_explicit_and_consumed_cursors_cannot_be_reused(self):
        client = fake_client()
        client.request.side_effect = [
            {"resourceType": "Bundle", "total": 2, "entry": [{"resource": observation("one", 1, "2025-01-01")}],
             "link": [{"relation": "next", "url": "?page=2"}]},
            {"resourceType": "Bundle", "total": 2, "entry": [{"resource": observation("two", 2, "2025-01-02")}]},
        ]
        tools = PatientTools(client, "Patient/p1")
        first = tools.execute("search_fhir", {"resource_type": "Observation", "count": 1})
        self.assertFalse(first["complete"])
        second = tools.execute("next_page", {"cursor": first["next_page"]})
        self.assertTrue(second["complete"])
        self.assertEqual(second["received"], 2)
        self.assertIn("/fhir/Observation?page=2", client.request.call_args.args[0])
        self.assertIn("error", tools.execute("next_page", {"cursor": first["next_page"]}))
        self.assertEqual(tools.http_requests, 2)

    def test_incomplete_or_wrong_patient_results_are_errors_and_never_evidence(self):
        for bundle in (
            {"resourceType": "Bundle", "total": 2, "entry": [{"resource": observation("one", 1, "2025-01-01")}]},
            {"resourceType": "Bundle", "total": 2, "entry": [{"resource": observation("one", 1, "2025-01-01")},
                {"resource": observation("wrong", 2, "2025-01-01", patient="p2")}]},
            {"resourceType": "Bundle", "entry": [{"search": {"mode": "outcome"},
                "resource": {"resourceType": "OperationOutcome"}}]},
        ):
            with self.subTest(bundle=bundle):
                client = fake_client()
                client.request.return_value, client.request.side_effect = bundle, None
                tools = PatientTools(client, "Patient/p1")
                result = tools.execute("search_fhir", {"resource_type": "Observation"})
                self.assertIn("error", result)
                self.assertFalse(result["complete"])
                self.assertFalse(tools.seen)

    def test_reference_reads_and_clinical_projections(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        self.assertIn("error", tools.execute("read_fhir", {"reference": "Medication/med-B"}))
        tools.execute("search_fhir", {"resource_type": "MedicationRequest"})
        result = tools.execute("read_fhir", {"reference": "Medication/med-B"})
        self.assertEqual(result["resource"]["code"]["coding"][0]["code"], "B")
        self.assertIn("Medication/med-B", tools.seen)
        self.assertIn("error", tools.execute("read_fhir", {"reference": "Patient/p2"}))
        tools.execute("read_fhir", {"reference": "Patient/p1"})
        self.assertIn("Patient/p1", tools.seen)

    def test_transport_errors_are_not_empty_results_and_interrupts_propagate(self):
        client = fake_client()
        tools = PatientTools(client, "Patient/p1")
        client.request.side_effect = URLError("offline")
        result = tools.execute("search_fhir", {"resource_type": "Observation"})
        self.assertIn("error", result)
        self.assertNotIn("resources", result)
        client.request.side_effect = InterruptedError("stop")
        with self.assertRaises(InterruptedError):
            tools.execute("search_fhir", {"resource_type": "Observation"})

    def test_clinical_projections_preserve_all_fixture_answers_and_evidence(self):
        resources = fixture()
        rows, _, _, _ = build_questions(Snapshot(resources), config())

        def rehydrate(original, projected):
            # v6 omits subject (patient scope is automatic) and shortens status concepts to
            # their code. Restore exactly that information, then re-derive every answer.
            projected = dict(projected)
            if "subject" in original:
                projected["subject"] = original["subject"]
            for key in ("clinicalStatus", "verificationStatus"):
                if isinstance(projected.get(key), str):
                    projected[key] = {"coding": [{"code": projected[key]}]}
            return projected

        projected = Snapshot([rehydrate(r, projection(r)) for r in resources if r["resourceType"] in FIELDS])
        self.assertEqual(len({r["family"] for r in rows}), 14)
        for row in rows:
            with self.subTest(family=row["family"]):
                answer, evidence = evaluate(projected, row["patient_id"], row["oracle_spec"])
                self.assertEqual(answer, row["ground_truth_answer"])
                self.assertEqual(sorted(evidence), row["ground_truth_resource_ids"])

    def test_v6_projection_is_slimmer_but_keeps_ids_references_and_names(self):
        condition = {"resourceType": "Condition", "id": "c1", "subject": {"reference": "Patient/p1"},
            "encounter": {"reference": "Encounter/e1"}, "meta": {"versionId": "1"},
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006", "display": "Diabetes"}], "text": "Diabetes"},
            "onsetDateTime": "2020-01-01T00:00:00+00:00"}
        projected = projection(condition)
        self.assertEqual(projected, {"resourceType": "Condition", "id": "c1", "encounter": {"reference": "Encounter/e1"},
            "clinicalStatus": "active", "verificationStatus": "confirmed",
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006", "display": "Diabetes"}]},
            "onsetDateTime": "2020-01-01T00:00:00+00:00"})
        # Text that differs from every display is information, so it stays.
        differing = dict(condition, code=dict(condition["code"], text="Type 2 diabetes"))
        self.assertEqual(projection(differing)["code"]["text"], "Type 2 diabetes")
        report = {"resourceType": "DiagnosticReport", "id": "r1", "subject": {"reference": "Patient/p1"},
            "issued": "2020-01-01T00:00:01.123+00:00", "effectiveDateTime": "2020-01-01T00:00:00+00:00",
            "result": [{"reference": "Observation/o1", "display": "Hemoglobin [Mass/volume] in Blood"}]}
        self.assertEqual(projection(report), {"resourceType": "DiagnosticReport", "id": "r1",
            "effectiveDateTime": "2020-01-01T00:00:00+00:00", "result": [{"reference": "Observation/o1"}]})

    def test_v6_page_size_cap(self):
        tools = PatientTools(fake_client(), "Patient/p1")
        self.assertEqual(TOOL_DEFINITIONS[0]["function"]["parameters"]["properties"]["count"]["maximum"], MAX_COUNT)
        result = tools.execute("search_fhir", {"resource_type": "Observation", "count": MAX_COUNT + 1})
        self.assertIn("next_page", result["error"])
        self.assertNotIn("error", tools.execute("search_fhir", {"resource_type": "Observation", "count": MAX_COUNT}))

    def test_strict_search_and_external_pagination_protection(self):
        client = FHIR("http://localhost:8080/fhir", strict_search=True)
        payload = {"resourceType": "Bundle", "total": 2,
            "entry": [{"resource": observation("one", 1, "2025-01-01")}],
            "link": [{"relation": "next", "url": "http://other.example/fhir/Observation?page=2"}]}
        with patch("dataset.fhir.urlopen", return_value=io.StringIO(json.dumps(payload))) as request:
            tools = PatientTools(client, "Patient/p1")
            result = tools.execute("search_fhir", {"resource_type": "Observation"})
            self.assertEqual(request.call_args.args[0].get_header("Prefer"), "handling=strict")
            self.assertIn("error", tools.execute("next_page", {"cursor": result["next_page"]}))
            self.assertEqual(request.call_count, 1)


class AgentTests(unittest.TestCase):
    def test_invalid_status_can_be_repaired_and_correct_answer_keeps_evidence(self):
        answer, evidence = evaluate(Snapshot(fixture()), "Patient/p1", {"operation": "cbc_panel", "as_of": "2026-10-02"})
        model = ScriptedModel([
            call("search_fhir", resource_type="DiagnosticReport", params={"status": "finished"}),
            call("search_fhir", resource_type="DiagnosticReport", params={"status": "final", "_id": "cbc1"}),
            call("read_fhir", reference="Observation/hb1") + call("read_fhir", reference="Observation/plt1"),
            call("finish", answer=answer, evidence=evidence),
        ])
        client = fake_client()
        episode = run_agent(CBC_QUESTION, "Patient/p1", model, client, Settings(), 0)
        self.assertEqual(episode["status"], "completed")
        self.assertEqual(episode["metrics"]["tool_errors"], 1)
        self.assertEqual(client.request.call_count, 3)
        self.assertEqual(set(episode["retrieved_resource_ids"]), set(evidence))
        self.assertIn("Invalid DiagnosticReport status", model.inputs[1][0][-1]["content"])
        self.assertTrue(score_episode({"ground_truth_answer": answer,
            "ground_truth_resource_ids": evidence}, episode)["answer_correct_with_evidence"])

    def test_visible_wording_selects_one_public_schema_for_all_fourteen_tasks(self):
        rows, _, _, _ = build_questions(Snapshot(fixture()), config())
        names = set()
        for row in rows:
            name, shape = contract_for_question(row["question"])
            names.add(name)
            with self.subTest(task=name):
                self.assertIsNone(answer_error(row["ground_truth_answer"], shape))
                schema = json_schema(shape)
                choices = schema.get("anyOf", [schema])
                self.assertTrue(all(s["additionalProperties"] is False for s in choices))
        self.assertEqual(len(names), 14)

    def test_wrong_task_answers_are_rejected_in_the_actual_agent_loop(self):
        question = ("For the most recent completed inpatient stay that contains at least one recorded "
                    "non-draft, non-error medication order and has a full 60-day window, compare products.")
        wrong = {"admitted_at": "2025-01-01T00:00:00Z", "discharged_at": "2025-01-02T00:00:00Z",
                 "previous_admitted_at": "2024-01-01T00:00:00Z",
                 "products": {"both": [], "only_first": [], "only_second": []}}
        # This was accepted by v3's union validator. It must not finish this question.
        self.assertIsNone(answer_error(wrong))
        corrected = {"admitted_at": wrong["admitted_at"], "discharged_at": wrong["discharged_at"], "followup": None}
        model = ScriptedModel([call("finish", answer=wrong, evidence=[]),
                               call("finish", answer=corrected, evidence=[])])
        episode = run_agent(question, "Patient/p1", model, fake_client(), Settings(), 0)
        self.assertEqual(episode["final"]["answer"], corrected)
        self.assertEqual(episode["metrics"]["protocol_errors"], 1)
        self.assertEqual(episode["answer_contract"], "Products at discharge follow-up")
        # v6: the selected schema appears once, in the user message; finish stays generic.
        schema = canonical(json_schema(contract_for_question(question)[1]))
        user = model.inputs[0][0][1]["content"]
        self.assertIn("Answer JSON Schema:\n" + schema, user)
        self.assertNotIn("previous_admitted_at", schema)
        finish_answer = model.inputs[0][1][-1]["function"]["parameters"]["properties"]["answer"]
        self.assertEqual(set(finish_answer), {"type", "description"})
        self.assertNotIn(schema, canonical(model.inputs[0][1]))

    def test_unknown_question_fails_before_model_or_http_instead_of_using_union(self):
        model, client = ScriptedModel([]), fake_client()
        with self.assertRaisesRegex(ValueError, "Unsupported or ambiguous"):
            run_agent("Unrecognized question", "Patient/p1", model, client, Settings(), 0)
        self.assertFalse(model.inputs)
        client.request.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Unsupported or ambiguous"):
                execute_run([{"id": "unknown", "split": "train", "question": "Unrecognized question"}],
                              client, model, Settings(), root / "output.jsonl", root)
            self.assertFalse((root / "output.jsonl").exists())

    def cbc_episode(self):
        snapshot = Snapshot(fixture())
        answer, evidence = evaluate(snapshot, "Patient/p1", {"operation": "cbc_panel", "as_of": "2026-10-02"})
        model = ScriptedModel([
            call("search_fhir", resource_type="DiagnosticReport", params={"_id": "cbc1"}),
            call("read_fhir", reference="Observation/hb1") + '\n' + call("read_fhir", reference="Observation/plt1"),
            call("finish", answer=answer, evidence=evidence),
        ])
        episode = run_agent(CBC_QUESTION + " QUESTION_SENTINEL", "Patient/p1", model, fake_client(), Settings(), 1)
        return model, episode, answer, evidence

    def test_native_tool_loop_observes_results_before_finishing(self):
        model, episode, answer, evidence = self.cbc_episode()
        self.assertEqual(episode["status"], "completed")
        self.assertEqual(episode["final"]["answer"], answer)
        self.assertEqual(episode["metrics"]["tool_calls"], 3)
        self.assertEqual(episode["metrics"]["llm_calls"], 3)
        self.assertTrue(set(evidence) <= set(episode["retrieved_resource_ids"]))
        observed = [json.loads(m["content"]) for m in model.inputs[2][0] if m["role"] == "tool"]
        self.assertEqual(observed[-1]["resource"]["id"], "plt1")
        expected_schema = json_schema(contract_for_question(CBC_QUESTION)[1])
        self.assertEqual(model.inputs[0][1], tool_definitions())
        self.assertIn(canonical(expected_schema), model.inputs[0][0][1]["content"])

    def test_calculator_result_reaches_model_and_counts_against_tool_budget(self):
        model = ScriptedModel([
            call("calculate", earlier="15.543", later="11.259", decimal_places=1),
            call("read_fhir", reference="Patient/p1"),
        ])
        client = fake_client()
        episode = run_agent(CBC_QUESTION, "Patient/p1", model, client,
                            Settings(max_tool_calls=1), 0)
        self.assertEqual(episode["status"], "tool_call_limit")
        self.assertEqual(episode["metrics"]["tool_calls"], 1)
        self.assertEqual(episode["metrics"]["http_requests"], 0)
        result = json.loads(model.inputs[1][0][-1]["content"])
        self.assertEqual(result["value"], "-4.3")
        self.assertFalse(episode["retrieved_resource_ids"])
        client.request.assert_not_called()

    def test_protocol_error_and_unseen_citation_can_be_repaired(self):
        answer = {"report_time": "2025-01-01T00:00:00Z", "hemoglobin": {"status": "no_matching_record"},
                  "platelets": {"status": "no_matching_record"}}
        model = ScriptedModel(["bad format", call("finish", answer={}, evidence=["Patient/p1"]),
            call("read_fhir", reference="Patient/p1"), canonical({"answer": answer, "evidence": ["Patient/p1"]})])
        episode = run_agent(CBC_QUESTION, "Patient/p1", model, fake_client(), Settings(), 3)
        self.assertEqual(episode["status"], "completed")
        self.assertEqual(episode["metrics"]["protocol_errors"], 2)
        self.assertEqual(episode["metrics"]["llm_calls"], 4)
        self.assertEqual(episode["metrics"]["tool_calls"], 1)

    def test_answer_schema_feedback_allows_repair_without_labels(self):
        answer = {"report_time": "2025-01-01T00:00:00Z",
                  "hemoglobin": {"status": "no_matching_record"},
                  "platelets": {"status": "no_matching_record"}}
        model = ScriptedModel([
            call("finish", answer={"latest_CBC": answer}, evidence=[]),
            call("finish", answer=answer, evidence=[]),
        ])
        episode = run_agent(CBC_QUESTION, "Patient/p1", model, fake_client(), Settings(), 0)
        self.assertEqual(episode["status"], "completed")
        self.assertEqual(episode["final"]["answer"], answer)
        self.assertEqual(episode["metrics"]["protocol_errors"], 1)
        feedback = [m for m in model.inputs[1][0] if m["role"] == "tool"][-1]
        self.assertIn("Answer schema error", feedback["content"])
        self.assertIn("wrapper", feedback["content"])
        self.assertEqual(episode["retrieved_resource_ids"], [])

    def test_public_contract_accepts_all_fixture_families_and_rejects_bad_measurements(self):
        rows, _, _, _ = build_questions(Snapshot(fixture()), config())
        self.assertEqual(len({r["family"] for r in rows}), 14)
        for row in rows:
            with self.subTest(family=row["family"]):
                self.assertIsNone(answer_error(row["ground_truth_answer"]))
        answer = copy.deepcopy(next(r["ground_truth_answer"] for r in rows if r["family"] == "cbc_panel"))
        for value in (True, "NaN", "Infinity", None):
            broken = copy.deepcopy(answer)
            broken["hemoglobin"]["value"] = value
            self.assertIn("value", answer_error(broken))
        broken = copy.deepcopy(answer)
        broken["hemoglobin"] = {"status": "missing_value", "value": None}
        self.assertIn("time", answer_error(broken))
        broken["hemoglobin"]["status"] = {"invalid": "status"}
        self.assertIsNotNone(answer_error(broken))
        self.assertIsNotNone(answer_error({}))

    def test_public_contract_enforces_conditional_followup_fields(self):
        stay = {"admitted_at": "2025-01-01T00:00:00Z", "discharged_at": "2025-01-02T00:00:00Z"}
        self.assertIsNone(answer_error(dict(stay, followup=None)))
        self.assertIsNotNone(answer_error(dict(stay, followup=None,
            products={"both": [], "only_first": [], "only_second": []})))
        concept = {"system": "example", "code": "x", "name": "Example"}
        glycemic = {"condition": concept, "onset": "2025-01-01T00:00:00Z",
                    "followup": {"status": "no_matching_record"}}
        self.assertIsNone(answer_error(glycemic))
        self.assertIsNotNone(answer_error(dict(glycemic, baseline={"status": "no_matching_record"})))

    def test_context_step_tool_and_output_limits_are_visible(self):
        cases = [
            ([ContextLimit("full")], Settings(), "context_limit"),
            (["bad"], Settings(max_steps=1), "step_limit"),
            (["bad"], Settings(max_output_tokens=10), "output_token_limit"),
            ([call("read_fhir", reference="Patient/p1") + call("read_fhir", reference="Patient/p1")],
             Settings(max_tool_calls=1), "tool_call_limit"),
            ([RuntimeError("model failed")], Settings(), "model_error"),
        ]
        for responses, settings, status in cases:
            with self.subTest(status=status):
                episode = run_agent(CBC_QUESTION, "Patient/p1", ScriptedModel(responses), fake_client(), settings, 0)
                self.assertEqual(episode["status"], status)
                self.assertIsNone(episode["final"])
                self.assertEqual(episode["metrics"]["llm_calls"], 1)

    def test_finish_cannot_precede_observation_in_same_tool_batch(self):
        with self.assertRaisesRegex(ValueError, "finish alone"):
            parse_calls(call("read_fhir", reference="Patient/p1") + call("finish", answer={}, evidence=[]))
        for value in ("NaN", "Infinity", "1e99999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_calls('{"answer":{"value":' + value + '},"evidence":[]}')

    def test_bare_tool_json_uses_the_same_strict_argument_checks(self):
        valid = {"name": "finish", "arguments": {"answer": {}, "evidence": []}}
        self.assertEqual(parse_calls(canonical(valid)), [valid])
        self.assertEqual(parse_calls("```json\n" + canonical(valid) + "\n```"), [valid])
        for invalid in (
            {"name": "finish", "arguments": "{}"},
            {"name": "finish", "arguments": {}, "extra": True},
            {"name": 1, "arguments": {}},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_calls(canonical(invalid))
        with self.assertRaises(ValueError):
            parse_calls(canonical(valid)[:-1])
        episode = run_agent(CBC_QUESTION, "Patient/p1", ScriptedModel([canonical(valid)]),
                            fake_client(), Settings(max_steps=1), 0)
        self.assertIsNone(episode["final"])
        self.assertEqual(episode["metrics"]["protocol_errors"], 1)

    def test_ambiguous_incomplete_answers_do_not_suggest_unrelated_tasks(self):
        for answer in (
            {"both": [], "only_first": [], "only_second": []},
            {"products": {"both": [], "only_first": [], "only_second": []}},
            {"renal_laboratory_follow_up": {"baseline": {"status": "no_matching_record"}}},
        ):
            with self.subTest(answer=answer):
                error = answer_error(answer)
                self.assertNotIn("Procedure indications", error)
                self.assertNotIn("closest", error)
                self.assertIn("question", error)
        error = answer_error({"procedures": []})
        self.assertIn("encounter_started_at", error)

    def test_repeated_failures_warn_then_stop_without_fabricating_an_answer(self):
        cases = [
            ("not json", fake_client(), "parse"),
            (call("finish", answer={"procedures": []}, evidence=[]), fake_client(), "schema"),
            (call("search_fhir", resource_type="Observation", params={"sort": "date"}), fake_client(), "tool"),
            (call("search_fhir", resource_type="Observation"), fake_client([]), "empty"),
        ]
        for response, client, kind in cases:
            with self.subTest(kind=kind):
                model = ScriptedModel([response] * 4)
                episode = run_agent(CBC_QUESTION, "Patient/p1", model, client, Settings(), 0)
                self.assertEqual(episode["status"], "repeated_failure")
                self.assertIsNone(episode["final"])
                self.assertEqual(episode["metrics"]["llm_calls"], 4)
                self.assertEqual(episode["metrics"]["repeat_warnings"], 1)
                feedback = model.inputs[3][0][-1]["content"]
                self.assertIn("next identical failure ends the episode", feedback)
                if kind == "empty":
                    self.assertEqual(episode["metrics"]["tool_errors"], 0)

    def test_new_evidence_resets_failure_guard_and_valid_reads_do_not_trigger_it(self):
        rejected = call("finish", answer={"procedures": []}, evidence=[])
        answer = {"report_time": "2025-01-01T00:00:00Z",
                  "hemoglobin": {"status": "no_matching_record"},
                  "platelets": {"status": "no_matching_record"}}
        model = ScriptedModel([rejected] * 3 + [call("read_fhir", reference="Patient/p1")]
                              + [rejected] * 3 + [call("finish", answer=answer, evidence=["Patient/p1"])])
        episode = run_agent(CBC_QUESTION, "Patient/p1", model, fake_client(), Settings(), 0)
        self.assertEqual(episode["status"], "completed")
        self.assertEqual(episode["metrics"]["repeat_warnings"], 2)
        model = ScriptedModel([call("read_fhir", reference="Patient/p1")] * 5
                              + [call("finish", answer=answer, evidence=["Patient/p1"])])
        episode = run_agent(CBC_QUESTION, "Patient/p1", model, fake_client(), Settings(), 0)
        self.assertEqual(episode["status"], "completed")
        self.assertEqual(episode["metrics"]["repeat_warnings"], 0)

    def test_output_budget_is_applied_before_each_generation(self):
        model = ScriptedModel(["bad", "bad"])
        episode = run_agent(CBC_QUESTION, "Patient/p1", model, fake_client(), Settings(max_output_tokens=25), 0)
        self.assertEqual(model.budgets, [25, 5])
        self.assertEqual(episode["metrics"]["output_tokens"], 25)
        self.assertEqual(episode["status"], "output_token_limit")

    def test_interrupt_is_not_swallowed_as_a_model_failure(self):
        with self.assertRaises(InterruptedError):
            run_agent(CBC_QUESTION, "Patient/p1", ScriptedModel([InterruptedError("stop")]), fake_client(), Settings(), 0)

    def test_model_gets_no_question_labels_and_output_is_one_file(self):
        rows, _, _, _ = build_questions(Snapshot(fixture()), config())
        row = copy.deepcopy(next(r for r in rows if r["family"] == "cbc_panel"))
        row.update(id="ID_SECRET_SENTINEL", split="train", family="FAMILY_SECRET_SENTINEL", required_operations=["OPERATIONS_SECRET_SENTINEL"],
                   oracle_spec={"SECRET_ORACLE": True}, question=CBC_QUESTION + " QUESTION_SENTINEL")
        answer, evidence = row["ground_truth_answer"], row["ground_truth_resource_ids"]
        model = ScriptedModel([call("search_fhir", resource_type="DiagnosticReport", params={"_id": "cbc1"}),
            call("read_fhir", reference="Observation/hb1") + call("read_fhir", reference="Observation/plt1"),
            call("finish", answer=answer, evidence=evidence)])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_questions(root / "questions.jsonl", [row])
            output = root / "output/zero-shot.jsonl"
            summary = execute_run([row], fake_client(), model, Settings(), output, root)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["overall"]["correct_with_evidence"], 1)
            self.assertEqual(list(output.parent.iterdir()), [output])
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r["type"] for r in records], ["run", "example", "summary"])
            self.assertEqual(records[0]["system_prompt"], SYSTEM_PROMPT)
            self.assertEqual(records[0]["interface_version"], INTERFACE_VERSION)
            self.assertIn("student/contract.py", records[0]["code_sha256"])
            contract = records[0]["public_answer_contracts"]["ID_SECRET_SENTINEL"]
            self.assertEqual(contract["name"], "Latest CBC")
            self.assertIn("Answer JSON Schema:\n" + canonical(contract["schema"]), model.inputs[0][0][1]["content"])
            first_prompt = canonical(model.inputs[0])
            self.assertIn("QUESTION_SENTINEL", first_prompt)
            for hidden in ("ID_SECRET_SENTINEL", "FAMILY_SECRET_SENTINEL", "OPERATIONS_SECRET_SENTINEL", "SECRET_ORACLE", "11.35"):
                self.assertNotIn(hidden, first_prompt)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                execute_run([row], fake_client(), model, Settings(), output, root)
            self.assertEqual(output.read_bytes(), before)
            row["split"] = "test"
            with self.assertRaisesRegex(ValueError, "training"):
                execute_run([row], fake_client(), model, Settings(), root / "other.jsonl", root)


class ScoringAndSelectionTests(unittest.TestCase):
    def test_semantic_comparison_handles_codes_order_numbers_units_and_times(self):
        expected = {"items": [{"system": "s", "code": "1", "name": "One"}, {"system": "s", "code": "2", "name": "Two"}],
                    "value": "1.2500", "unit": "mm[Hg]", "time": "2025-01-01T00:00:00+00:00"}
        actual = {"items": [{"system": "s", "code": "2", "name": "Different display"}, {"system": "s", "code": "1"}],
                  "value": 1.25, "unit": "mmHg", "time": "2024-12-31T19:00:00-05:00"}
        self.assertTrue(equal(actual, expected))
        actual["value"] = 1.3
        self.assertFalse(equal(actual, expected))
        self.assertFalse(equal([1], [1, 1]))
        self.assertFalse(equal({"status": "missing_value", "value": None}, {"status": "no_matching_record"}))
        self.assertFalse(equal(True, "1", "value"))
        self.assertFalse(equal("NaN", "1", "value"))
        self.assertFalse(equal("1e99999999999999999999", "1", "value"))

    def test_answer_accuracy_and_evidence_coverage_are_separate(self):
        row = {"ground_truth_answer": {"value": "1"}, "ground_truth_resource_ids": ["Observation/a", "Encounter/b"]}
        episode = {"final": {"answer": {"value": 1}, "evidence": ["Encounter/b"]},
                   "retrieved_resource_ids": ["Encounter/b", "Observation/a"]}
        scored = score_episode(row, episode)
        self.assertTrue(scored["answer_correct"])
        self.assertFalse(scored["answer_correct_with_evidence"])
        self.assertEqual(scored["citation_evidence_recall"], 0.5)
        episode["final"]["evidence"].append("Observation/a")
        self.assertTrue(score_episode(row, episode)["answer_correct_with_evidence"])

    def test_balanced_selection_is_stable_train_only_and_does_not_mutate_rows(self):
        original, _, _, _ = build_questions(Snapshot(fixture()), config())
        rows = []
        for row in original:
            for i in range(5):
                copy_row = copy.deepcopy(row)
                copy_row.update(id=f"{row['id']}-{i}", split="train" if i < 4 else "test")
                if row["family"] == "discharge_followup" and i == 0:
                    copy_row["ground_truth_answer"]["followup"] = None
                rows.append(copy_row)
        before = copy.deepcopy(rows)
        selected = select_questions(rows)
        self.assertEqual(len(selected), 56)
        self.assertTrue(all(r["split"] == "train" for r in selected))
        self.assertEqual(selected, select_questions(list(reversed(rows))))
        self.assertEqual(rows, before)
        summary = selection_summary(selected)
        self.assertEqual(summary["families"]["discharge_followup"]["negative"], 1)
        with self.assertRaises(ValueError):
            select_questions(rows, 5)

    def test_callback_runs_inside_server_lifetime_and_failure_still_cleans_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = Snapshot(fixture())
            rows, _, _, _ = build_questions(snapshot, config())
            rows = [r for r in rows if r["patient_id"] == "Patient/p1"]
            write_questions(root / "questions.jsonl", rows)
            (root / "main.war").write_bytes(b"test-only war")
            process = MagicMock()
            process.poll.return_value = None
            result = SimpleNamespace(returncode=0, stdout="", stderr="test")
            artifact = {"war_sha256": file_digest(root / "main.war")}
            args = SimpleNamespace(output=root / "live", dataset=root, snapshot=root, batch_size=100)
            report = {"valid": True, "questions_checked": len(rows), "patients_checked": 1}
            def callback(client, selected, source):
                process.terminate.assert_not_called()
                self.assertEqual(selected, rows)
                self.assertIs(source, snapshot)
                raise ValueError("agent callback failed")
            with patch("dataset.live.artifact_info", return_value=(root / "main.war", artifact)), \
                 patch("dataset.live.shutil.which", return_value="/mock/bin"), \
                 patch("dataset.live.subprocess.run", return_value=result) as commands, \
                 patch("dataset.live.subprocess.check_output", return_value="test"), \
                 patch("dataset.live.subprocess.Popen", return_value=process), \
                 patch("dataset.live.free_port", side_effect=[15432, 18080]), \
                 patch("dataset.live.Snapshot.load", return_value=snapshot), \
                 patch("dataset.live.validate", return_value={"valid": True}), \
                 patch("dataset.live.wait_ready", return_value={"resourceType": "CapabilityStatement"}), \
                 patch("dataset.live.import_snapshot"), \
                 patch("dataset.live.verify_dataset", return_value=report):
                with self.assertRaisesRegex(ValueError, "agent callback failed"):
                    run_job(args, rows=rows, after_verify=callback)
            process.terminate.assert_called_once()
            self.assertTrue(any(c.args[0][0] == "pg_ctl" and c.args[0][-1] == "stop" for c in commands.call_args_list))
            self.assertFalse(json.loads((root / "live/status.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
