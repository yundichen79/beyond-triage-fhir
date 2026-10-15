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
from pilot.agent import ContextLimit, Generation, Settings, SYSTEM_PROMPT, parse_calls, run_agent
from pilot.run import execute_pilot, select_questions, selection_summary, summarize
from pilot.scoring import equal, score_episode
from pilot.tools import FIELDS, PatientTools, TOOL_DEFINITIONS, projection
from test_dataset import fixture, config, observation


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
        parsed = urlparse(path)
        kind = parsed.path.rstrip("/").split("/")[-1]
        params = parse_qs(parsed.query)
        resources = list(snapshot.for_patient(params["patient"][0], kind))
        if "_id" in params:
            resources = [r for r in resources if r["id"] == params["_id"][0]]
        return {"resourceType": "Bundle", "total": len(resources),
                "entry": [{"resource": copy.deepcopy(r)} for r in resources]}
    client.request = MagicMock(side_effect=request)
    return client


class ToolTests(unittest.TestCase):
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
        projected = Snapshot([projection(r) for r in resources if r["resourceType"] in FIELDS])
        self.assertEqual(len({r["family"] for r in rows}), 14)
        for row in rows:
            with self.subTest(family=row["family"]):
                answer, evidence = evaluate(projected, row["patient_id"], row["oracle_spec"])
                self.assertEqual(answer, row["ground_truth_answer"])
                self.assertEqual(sorted(evidence), row["ground_truth_resource_ids"])

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
    def cbc_episode(self):
        snapshot = Snapshot(fixture())
        answer, evidence = evaluate(snapshot, "Patient/p1", {"operation": "cbc_panel", "as_of": "2026-10-02"})
        model = ScriptedModel([
            call("search_fhir", resource_type="DiagnosticReport", params={"_id": "cbc1"}),
            call("read_fhir", reference="Observation/hb1") + '\n' + call("read_fhir", reference="Observation/plt1"),
            call("finish", answer=answer, evidence=evidence),
        ])
        episode = run_agent("QUESTION_SENTINEL", "Patient/p1", model, fake_client(), Settings(), 1)
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
        self.assertEqual(model.inputs[0][1], TOOL_DEFINITIONS)

    def test_protocol_error_and_unseen_citation_can_be_repaired(self):
        model = ScriptedModel(["bad format", call("finish", answer={}, evidence=["Patient/p1"]),
            call("read_fhir", reference="Patient/p1"), canonical({"answer": {}, "evidence": ["Patient/p1"]})])
        episode = run_agent("Question", "Patient/p1", model, fake_client(), Settings(), 3)
        self.assertEqual(episode["status"], "completed")
        self.assertEqual(episode["metrics"]["protocol_errors"], 2)
        self.assertEqual(episode["metrics"]["llm_calls"], 4)
        self.assertEqual(episode["metrics"]["tool_calls"], 1)

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
                episode = run_agent("Q", "Patient/p1", ScriptedModel(responses), fake_client(), settings, 0)
                self.assertEqual(episode["status"], status)
                self.assertIsNone(episode["final"])
                self.assertEqual(episode["metrics"]["llm_calls"], 1)

    def test_finish_cannot_precede_observation_in_same_tool_batch(self):
        with self.assertRaisesRegex(ValueError, "finish alone"):
            parse_calls(call("read_fhir", reference="Patient/p1") + call("finish", answer={}, evidence=[]))
        for value in ("NaN", "Infinity", "1e99999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_calls('{"answer":{"value":' + value + '},"evidence":[]}')

    def test_output_budget_is_applied_before_each_generation(self):
        model = ScriptedModel(["bad", "bad"])
        episode = run_agent("Q", "Patient/p1", model, fake_client(), Settings(max_output_tokens=25), 0)
        self.assertEqual(model.budgets, [25, 5])
        self.assertEqual(episode["metrics"]["output_tokens"], 25)
        self.assertEqual(episode["status"], "output_token_limit")

    def test_interrupt_is_not_swallowed_as_a_model_failure(self):
        with self.assertRaises(InterruptedError):
            run_agent("Q", "Patient/p1", ScriptedModel([InterruptedError("stop")]), fake_client(), Settings(), 0)

    def test_model_gets_no_question_labels_and_output_is_one_file(self):
        rows, _, _, _ = build_questions(Snapshot(fixture()), config())
        row = copy.deepcopy(next(r for r in rows if r["family"] == "cbc_panel"))
        row.update(id="ID_SECRET_SENTINEL", split="train", family="FAMILY_SECRET_SENTINEL", required_operations=["OPERATIONS_SECRET_SENTINEL"],
                   oracle_spec={"SECRET_ORACLE": True}, question="QUESTION_SENTINEL")
        answer, evidence = row["ground_truth_answer"], row["ground_truth_resource_ids"]
        model = ScriptedModel([call("search_fhir", resource_type="DiagnosticReport", params={"_id": "cbc1"}),
            call("read_fhir", reference="Observation/hb1") + call("read_fhir", reference="Observation/plt1"),
            call("finish", answer=answer, evidence=evidence)])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_questions(root / "questions.jsonl", [row])
            output = root / "output/pilot.jsonl"
            summary = execute_pilot([row], fake_client(), model, Settings(), output, root)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["overall"]["correct_with_evidence"], 1)
            self.assertEqual(list(output.parent.iterdir()), [output])
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r["type"] for r in records], ["run", "example", "summary"])
            self.assertEqual(records[0]["system_prompt"], SYSTEM_PROMPT)
            first_prompt = canonical(model.inputs[0])
            self.assertIn("QUESTION_SENTINEL", first_prompt)
            for hidden in ("ID_SECRET_SENTINEL", "FAMILY_SECRET_SENTINEL", "OPERATIONS_SECRET_SENTINEL", "SECRET_ORACLE", "11.35"):
                self.assertNotIn(hidden, first_prompt)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                execute_pilot([row], fake_client(), model, Settings(), output, root)
            self.assertEqual(output.read_bytes(), before)
            row["split"] = "test"
            with self.assertRaisesRegex(ValueError, "training"):
                execute_pilot([row], fake_client(), model, Settings(), root / "other.jsonl", root)


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
                raise ValueError("pilot callback failed")
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
                with self.assertRaisesRegex(ValueError, "pilot callback failed"):
                    run_job(args, rows=rows, after_verify=callback)
            process.terminate.assert_called_once()
            self.assertTrue(any(c.args[0][0] == "pg_ctl" and c.args[0][-1] == "stop" for c in commands.call_args_list))
            self.assertFalse(json.loads((root / "live/status.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
