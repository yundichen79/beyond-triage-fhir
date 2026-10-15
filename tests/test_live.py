"""Import completeness, HTTP-derived answers, and server lifecycle regressions."""

import copy
import json
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dataset.common import SQLiteResources, Snapshot, canonical, file_digest, patient_ref, write_json, write_questions
from dataset.fhir import FHIR, check_import_search, import_snapshot, verify_dataset
from dataset.live import IMAGE, artifact_info, verification_output, paging_preflight_patient, run_job, wait_ready, war_launcher
from dataset.questions import build_questions
from dataset.scope import scope_for_patients
from test_dataset import config, fixture, observation


class LiveTests(unittest.TestCase):
    def test_import_probe_accepts_matching_clinical_content(self):
        resource = observation("probe", 1, "2025-01-01")
        client = FHIR("http://localhost:8080/fhir")
        found = dict(resource, meta={"versionId": "1"}, text={"status": "generated"})
        with patch.object(client, "search_url", return_value=[found]) as search:
            check_import_search(client, "Observation/probe", resource)
        self.assertIn("patient=Patient%2Fp1", search.call_args.args[0])
        self.assertIn("_id=probe", search.call_args.args[0])

    def test_import_probe_diagnoses_reads_and_reference_search_without_accepting_empty(self):
        resource = observation("probe", 1, "2025-01-01")
        client = FHIR("http://localhost:8080/fhir")
        with patch.object(client, "search_url", return_value=[]), patch.object(client, "request", side_effect=[
            resource, {"resourceType": "Bundle", "total": 1, "entry": [{"resource": resource}]},
            {"resourceType": "Bundle", "total": 0},
        ]):
            with self.assertRaisesRegex(ValueError, "Imported-record search probe failed") as error:
                check_import_search(client, "Observation/probe", resource)
        details = json.loads(str(error.exception).split(": ", 1)[1])
        self.assertTrue(details["read"]["matches_source"])
        self.assertEqual(details["by_id"]["total"], 1)
        self.assertEqual(details["bare_patient_id"]["total"], 0)
        self.assertEqual(client.timeout, 180)

    def test_live_import_stops_after_first_unsearchable_chunk(self):
        resources = [{"resourceType": "Patient", "id": "p1"},
                     observation("probe", 1, "2025-01-01"), observation("later", 2, "2025-01-02")]
        client = FHIR("http://localhost:8080/fhir")
        transactions = []
        def request(path="", body=None):
            if path == "metadata":
                return {"resourceType": "CapabilityStatement", "fhirVersion": "4.0.1"}
            transactions.append(body)
            return {"type": "transaction-response", "entry": [
                {"response": {"status": "201 Created"}} for _ in body["entry"]]}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(client, "request", side_effect=request), \
             patch.object(client, "search_all", return_value=[]), \
             patch("dataset.fhir.check_import_search", side_effect=ValueError("unsearchable")):
            with self.assertRaisesRegex(ValueError, "unsearchable"):
                import_snapshot(Snapshot(resources), client, Path(temporary) / "import.json", 2, probe_search=True)
        self.assertEqual(len(transactions), 1)

    def test_default_verification_output_is_removed_even_on_failure(self):
        with self.assertRaisesRegex(ValueError, "simulated"):
            with verification_output() as path:
                directory = path
                path.mkdir()
                write_json(path / "status.json", {"status": "failed", "passed": False, "error": "simulated"})
                raise ValueError("simulated")
        self.assertFalse(directory.exists())
        self.assertFalse(directory.parent.exists())
        with tempfile.TemporaryDirectory() as temporary:
            explicit = Path(temporary) / "requested"
            with verification_output(explicit) as output:
                output.mkdir()
            self.assertTrue(explicit.exists())

    def test_pagination_follows_more_than_eleven_pages(self):
        client = FHIR("http://127.0.0.1:8080/fhir")
        from urllib.parse import parse_qs, urlparse

        def request(path="", body=None):
            query = parse_qs(urlparse(path).query)
            offset = int(query.get("offset", [0])[0])
            bundle = {"resourceType": "Bundle", "total": 2401,
                      "entry": [{"resource": {"resourceType": "Observation", "id": str(index)}}
                                for index in range(offset, min(offset + 200, 2401))]}
            if offset + 200 < 2401:
                bundle["link"] = [{"relation": "next", "url": f"Observation?offset={offset + 200}"}]
            return bundle

        with patch.object(client, "request", side_effect=request) as calls:
            resources = client.search_all("Observation", "Patient/p1")
        self.assertEqual(len(resources), 2401)
        self.assertEqual(calls.call_count, 13)

    def test_search_rejects_silently_truncated_results(self):
        client = FHIR("http://127.0.0.1:8080/fhir")
        truncated = {"resourceType": "Bundle", "total": 2401,
                     "entry": [{"resource": {"resourceType": "Observation", "id": str(i)}} for i in range(2200)]}
        with patch.object(client, "request", return_value=truncated):
            with self.assertRaisesRegex(ValueError, "received 2200 of 2401"):
                client.search_all("Observation", "Patient/p1")

    def test_preflight_uses_largest_selected_patient_and_skips_small_charts(self):
        snapshot = Snapshot(fixture() + [observation(f"p2-extra-{i}", 50, "2025-01-01", patient="p2") for i in range(50)])
        self.assertEqual(paging_preflight_patient(snapshot, [{"patient_id": "Patient/p1"}], boundary=2), "Patient/p1")
        self.assertEqual(paging_preflight_patient(snapshot, [{"patient_id": "Patient/p1"}, {"patient_id": "Patient/p2"}], boundary=2), "Patient/p2")
        self.assertIsNone(paging_preflight_patient(snapshot, [{"patient_id": "Patient/p1"}], boundary=1000))

    def test_sqlite_scope_streams_the_same_complete_records_as_memory_scope(self):
        memory = Snapshot(fixture() + [observation("other", 50, "2025-01-01", patient="p2")])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE resources (key TEXT PRIMARY KEY, resource_type TEXT, patient TEXT, body TEXT)")
                connection.executemany("INSERT INTO resources VALUES (?,?,?,?)", [
                    (key, resource["resourceType"], patient_ref(resource), canonical(resource))
                    for key, resource in memory.resources.items()])
            disk = Snapshot.__new__(Snapshot)
            disk.id, disk.patients, disk.resources = memory.id, memory.patients, SQLiteResources(path)
            expected = scope_for_patients(memory, ["Patient/p1"])
            actual = scope_for_patients(disk, ["Patient/p1"])
            items = list(actual.resources.items())
            self.assertEqual(actual.scope, expected.scope)
            self.assertEqual(len(items), len(actual.resources))
            self.assertEqual(dict(items), dict(expected.resources.items()))
            # Loaded patient records support joins/evidence checks without another
            # disk query, and switching patients releases the previous cache.
            disk.for_patient("Patient/p1", "Observation")
            queries = []
            disk.resources.connection.set_trace_callback(queries.append)
            self.assertEqual(disk.resources["Observation/old"], memory.resources["Observation/old"])
            self.assertEqual(queries, [])
            disk.for_patient("Patient/p2", "Observation")
            self.assertNotIn("Observation/old", disk.resources.patient_cache)
            queries.clear()
            self.assertEqual(disk.resources["Observation/old"], memory.resources["Observation/old"])
            self.assertTrue(queries)

    def test_scope_includes_uncited_records_and_transitive_dependencies(self):
        resources = fixture() + [
            observation("uncited", 69, "2000-01-01"),
            observation("other-patient", 50, "2025-01-01", patient="p2"),
            {"resourceType": "Location", "id": "location", "managingOrganization": {"reference": "Organization/parent"}},
            {"resourceType": "Organization", "id": "parent", "name": "Parent hospital"},
        ]
        resources[8]["location"] = [{"location": {"reference": "Location/location"}}]
        scope = scope_for_patients(Snapshot(resources), ["Patient/p1"])
        self.assertIn("Observation/uncited", scope.resources)
        self.assertIn("Organization/parent", scope.resources)
        self.assertIn("Location/location", scope.resources)
        self.assertNotIn("Observation/other-patient", scope.resources)
        self.assertNotIn("Patient/p2", scope.resources)
        self.assertEqual(scope.patients, ["Patient/p1"])
        self.assertEqual(scope.scope["resource_count"], len(resources) - 2)

    def test_scope_rejects_missing_targets_and_cross_patient_references(self):
        resources = fixture()
        resources[0]["link"] = [{"other": {"reference": "Patient/p2"}, "type": "seealso"}]
        with self.assertRaisesRegex(ValueError, "unselected patient"):
            scope_for_patients(Snapshot(resources), ["Patient/p1"])
        resources[0]["link"][0]["other"]["reference"] = "Patient/missing"
        with self.assertRaisesRegex(ValueError, "Unresolved import reference"):
            scope_for_patients(Snapshot(resources), ["Patient/p1"])

    def test_war_launcher_and_artifact_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            war = root / "main.war"
            war.parent.mkdir(parents=True, exist_ok=True)
            for launcher in ("org.springframework.boot.loader.WarLauncher",
                             "org.springframework.boot.loader.launch.WarLauncher"):
                with zipfile.ZipFile(war, "w") as archive:
                    # Reproduce the real image: multiple loader generations coexist.
                    for available in ("org.springframework.boot.loader.WarLauncher",
                                      "org.springframework.boot.loader.launch.WarLauncher",
                                      "org.springframework.boot.loader.PropertiesLauncher"):
                        archive.writestr(available.replace(".", "/") + ".class", b"test fixture")
                    archive.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\r\nMain-Class: " + launcher
                                     + "\r\nStart-Class: example.Application\r\n\r\n")
                    archive.writestr("WEB-INF/classes/example/Application.class", b"test fixture")
                self.assertEqual(war_launcher(war), launcher)
                # Already downloaded artifacts retain their preparation provenance.
                write_json(root / "artifact.json", {"image": IMAGE, "war_sha256": file_digest(war),
                                                     "launcher": "org.springframework.boot.loader.PropertiesLauncher"})
                self.assertEqual(artifact_info(root)[0], war)
                self.assertEqual(artifact_info(root)[1]["launcher"], launcher)
            war.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity/checksum"):
                artifact_info(root)

    def test_war_rejects_missing_declared_application_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            war = Path(temporary) / "main.war"
            with zipfile.ZipFile(war, "w") as archive:
                archive.writestr("org/springframework/boot/loader/launch/WarLauncher.class", b"test fixture")
                archive.writestr("META-INF/MANIFEST.MF", "Main-Class: org.springframework.boot.loader.launch.WarLauncher\n"
                                 "Start-Class: example.Missing\n\n")
            with self.assertRaisesRegex(ValueError, "declared application class is missing"):
                war_launcher(war)

    def test_live_verifier_rejects_wrong_labels_even_with_matching_server(self):
        snapshot = Snapshot(fixture())
        rows, _, _, _ = build_questions(snapshot, config())
        rows = [r for r in rows if r["family"] == "cbc_panel" and r["patient_id"] == "Patient/p1"]
        client = FHIR("http://127.0.0.1:8080/fhir")

        def request(path="", body=None):
            if path == "metadata":
                return {"resourceType": "CapabilityStatement", "fhirVersion": "4.0.1"}
            return copy.deepcopy(snapshot.resources[path])

        with patch.object(client, "request", side_effect=request), patch.object(
                client, "search_all", side_effect=lambda rtype, patient: snapshot.for_patient(patient, rtype)):
            report = verify_dataset(snapshot, rows, client)
            self.assertTrue(report["valid"])
            self.assertEqual(report["questions_checked"], 1)
            rows[0]["ground_truth_answer"] = "1900-01-01"
            report = verify_dataset(snapshot, rows, client)
            self.assertFalse(report["valid"])
            self.assertEqual(report["questions_checked"], 0)
            self.assertIn("Live answer/evidence mismatch", report["errors"][0]["error"])
            self.assertFalse(verify_dataset(snapshot, [], client)["valid"])

    def test_wait_ready_reports_server_exit_without_waiting_for_timeout(self):
        process = MagicMock()
        process.poll.return_value = 1
        process.returncode = 1
        with self.assertRaisesRegex(ValueError, "exited before readiness"):
            wait_ready(MagicMock(), process)

    def test_import_failure_stops_both_servers_and_preserves_failure_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, _, _, _ = build_questions(Snapshot(fixture()), config())
            write_questions(root / "questions.jsonl", [r for r in rows if r["patient_id"] == "Patient/p1"])
            args = SimpleNamespace(output=root / "reports", dataset=root, snapshot=root, batch_size=100)
            (root / "main.war").write_bytes(b"mock archive for lifecycle test")
            result = SimpleNamespace(returncode=0, stdout="", stderr="java test version")
            process = MagicMock()
            process.poll.return_value = None
            artifact = {"launcher": "org.springframework.boot.loader.launch.WarLauncher", "image": IMAGE,
                        "war_sha256": file_digest(root / "main.war")}
            with patch("dataset.live.artifact_info", return_value=(root / "main.war", artifact)), \
                    patch("dataset.live.shutil.which", return_value="/mock/bin"), \
                    patch("dataset.live.subprocess.run", return_value=result) as commands, \
                    patch("dataset.live.subprocess.check_output", return_value="PostgreSQL 16.4"), \
                    patch("dataset.live.subprocess.Popen", return_value=process), \
                    patch("dataset.live.free_port", side_effect=[15432, 18080]), \
                    patch("dataset.live.Snapshot.load", return_value=Snapshot(fixture())), \
                    patch("dataset.live.validate", return_value={"valid": True}), \
                    patch("dataset.live.wait_ready", return_value={"resourceType": "CapabilityStatement"}), \
                    patch("dataset.live.import_snapshot", side_effect=ValueError("import failed")):
                with self.assertRaisesRegex(ValueError, "import failed"):
                    run_job(args)
            process.terminate.assert_called_once()
            self.assertTrue(any(call.args[0][0] == "pg_ctl" and call.args[0][-1] == "stop"
                                for call in commands.call_args_list))
            report = json.loads((root / "reports/status.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"], "import failed")
            self.assertTrue(report["postgres_stopped"])


if __name__ == "__main__":
    unittest.main()
