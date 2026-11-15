"""Regression checks for the CPU integration probe, including forced pagination."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock
from urllib.parse import parse_qsl, urlencode, urlparse

from dataset.common import Snapshot
from dataset.fhir import FHIR
from dataset.questions import build_questions
from pilot.smoke import check_queries, select_patient_rows
from fhir_fixture import search
from test_dataset import fixture, config


def paginated_client(snapshot):
    pages = {}
    client = FHIR("http://localhost:8080/fhir")

    def request(path):
        if path in pages:
            return copy.deepcopy(pages[path])
        if "?" not in path:
            return copy.deepcopy(snapshot.resources[path])
        parsed = urlparse(path)
        params = dict(parse_qsl(parsed.query))
        count = int(params.get("_count", 50))
        # Preserve repeated filter parameters; only override count for fixture evaluation.
        query = [(k, v) for k, v in parse_qsl(parsed.query) if k != "_count"] + [("_count", "1000000")]
        all_results = search(snapshot, parsed.path + "?" + urlencode(query))
        entries = all_results["entry"]
        chunks = [entries[i:i + count] for i in range(0, len(entries), count)] or [[]]
        urls = [f"{client.base}/{parsed.path}?fixture-page={len(pages) + i + 1}" for i in range(len(chunks))]
        bundles = []
        for i, chunk in enumerate(chunks):
            bundle = {"resourceType": "Bundle", "total": len(entries), "entry": chunk}
            if i + 1 < len(chunks):
                bundle["link"] = [{"relation": "next", "url": urls[i + 1]}]
            bundles.append(bundle)
            pages[urls[i]] = bundle
        return copy.deepcopy(bundles[0])

    client.request = MagicMock(side_effect=request)
    return client


class SmokeTests(unittest.TestCase):
    def setUp(self):
        resources = fixture()
        for resource in resources:
            if resource.get("resourceType") == "Condition" and resource.get("onsetDateTime"):
                resource["clinicalStatus"] = {"coding": [{"code": "active"}]}
        self.snapshot = Snapshot(resources)
        self.rows = [dict(row, split="train") for row in build_questions(self.snapshot, config())[0]]

    def test_selection_uses_only_training_patients_and_requires_coverage(self):
        self.assertEqual({r["patient_id"] for r in select_patient_rows(self.rows)}, {"Patient/p1"})
        with self.assertRaisesRegex(ValueError, "No training patient"):
            select_patient_rows([dict(r, split="test") for r in self.rows])

    def test_probe_compares_filters_and_sorting_and_follows_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            report = check_queries(paginated_client(self.snapshot), self.rows, self.snapshot, Path(directory))
            self.assertTrue(report["passed"])
            self.assertFalse(report["model_generation"])
            self.assertEqual(report["checks_passed"], report["checks_total"])
            checks = {c["name"]: c for c in report["checks"]}
            self.assertEqual(checks["finished-inpatient-sort"]["pages"], 2)
            self.assertEqual(checks["exclusive-lower-date-bound"]["actual_ids"], ["Observation/cr-followup"])
            self.assertEqual(checks["valid-empty-query"]["actual_ids"], [])
            self.assertEqual(checks["unseen-encounter"]["http_requests"], 0)
            self.assertTrue(json.loads((Path(directory) / "query-smoke.json").read_text())["passed"])

    def test_probe_records_failure_and_never_marks_bad_ordering_passed(self):
        client = paginated_client(self.snapshot)
        request = client.request.side_effect

        def wrong_order(path):
            if "class=IMP" in path and "_sort=-date" in path:
                encounters = [r for r in self.snapshot.for_patient("Patient/p1", "Encounter")
                              if r.get("class", {}).get("code") == "IMP"]
                encounters.sort(key=lambda r: r["period"]["start"])
                return {"resourceType": "Bundle", "total": len(encounters),
                        "entry": [{"resource": r} for r in encounters]}
            return request(path)

        client.request.side_effect = wrong_order
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, "ordering differs"):
                check_queries(client, self.rows, self.snapshot, Path(directory))
            saved = json.loads((Path(directory) / "query-smoke.json").read_text())
            self.assertFalse(saved["passed"])
            self.assertIn("ordering differs", saved["error"])
