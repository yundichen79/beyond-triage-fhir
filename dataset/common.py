"""Shared serialization and FHIR indexing primitives."""

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sequence_digest(values):
    """Hash a canonical JSON array incrementally, without materializing it."""
    h = hashlib.sha256(b"[")
    first = True
    for value in values:
        if not first:
            h.update(b",")
        h.update(canonical(value).encode())
        first = False
    h.update(b"]")
    return h.hexdigest()


def jsonl_digests(path):
    """Hash both a canonical JSONL file and its JSON-array identity in one pass."""
    file_hash, array_hash = hashlib.sha256(), hashlib.sha256(b"[")
    first = True
    with Path(path).open("rb") as handle:
        for line in handle:
            file_hash.update(line)
            if not line.strip():
                continue
            if not first:
                array_hash.update(b",")
            array_hash.update(line.rstrip(b"\r\n"))
            first = False
    array_hash.update(b"]")
    return file_hash.hexdigest(), array_hash.hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    return list(iter_jsonl(path))


def iter_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


QUESTION_FIELDS = (
    "id", "patient_id", "question", "family", "complexity",
    "required_operations", "split", "ground_truth_answer", "ground_truth_resource_ids",
    "answer_type", "oracle_spec",
)
def write_questions(path, rows):
    write_jsonl(path, rows)


def iter_questions(path):
    for row in iter_jsonl(path):
        if set(row) != set(QUESTION_FIELDS):
            raise ValueError("Question fields do not match the JSONL schema")
        if not isinstance(row["ground_truth_answer"], dict):
            raise ValueError("Clinical ground truth must be a structured object")
        if not isinstance(row["oracle_spec"], dict):
            raise ValueError("Question computation rules must be an object")
        for field in ("required_operations", "ground_truth_resource_ids"):
            if not isinstance(row[field], list) or any(not isinstance(x, str) for x in row[field]):
                raise ValueError(f"Expected a list of strings: {field}")
        yield row


def read_questions(path):
    return list(iter_questions(path))


def iter_agent_inputs(directory, split=None):
    """Yield only the fields safe to expose to an answering agent."""
    if split is not None and split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    for row in iter_questions(Path(directory) / "questions.jsonl"):
        if split is None or row["split"] == split:
            yield {key: row[key] for key in ("id", "question", "patient_id", "split")}


def resource_id(resource):
    return f"{resource['resourceType']}/{resource['id']}"


def patient_ref(resource):
    return (resource.get("subject") or resource.get("patient") or {}).get("reference")


def codes(concept):
    return {(c.get("system", ""), c.get("code", "")) for c in concept.get("coding", [])}


def timestamp(value):
    # Partial dates do not support unambiguous latest/within-year comparisons.
    if not isinstance(value, str) or len(value) < 10:
        raise ValueError(f"Missing or partial timestamp: {value!r}")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def observation_time(resource):
    # Do not substitute issued: it is publication time, not measurement time.
    return timestamp(resource.get("effectiveDateTime") or resource.get("effectivePeriod", {}).get("start"))


class Snapshot:
    def __init__(self, resources, snapshot_id="test"):
        self.id = snapshot_id
        self.resources = {}
        self.by_patient = defaultdict(lambda: defaultdict(list))
        for resource in resources:
            key = resource_id(resource)
            if key in self.resources:
                raise ValueError(f"Duplicate resource: {key}")
            self.resources[key] = resource
            ref = patient_ref(resource)
            if ref:
                self.by_patient[ref][resource["resourceType"]].append(resource)
        self.patients = sorted(k for k in self.resources if k.startswith("Patient/"))

    def for_patient(self, patient, resource_type):
        if isinstance(self.resources, SQLiteResources):
            # Retain only the current patient's relevant types, never a whole cohort.
            if getattr(self, "_cached_patient", None) != patient:
                self._cached_patient, self._cached_resources = patient, {}
                self.resources.patient_cache = {}
            if resource_type not in self._cached_resources:
                self._cached_resources[resource_type] = [
                    json.loads(row[0]) for row in self.resources.connection.execute(
                        "SELECT body FROM resources WHERE patient=? AND resource_type=? ORDER BY key",
                        (patient, resource_type))]
                self.resources.patient_cache.update({resource_id(resource): resource
                                                     for resource in self._cached_resources[resource_type]})
            return self._cached_resources[resource_type]
        return self.by_patient[patient][resource_type]

    @classmethod
    def load(cls, directory, on_progress=None):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        resources_path = directory / "resources.jsonl"
        if on_progress:
            on_progress("Verifying snapshot export checksum and identity")
        resources_hash, snapshot_hash = jsonl_digests(resources_path)
        if resources_hash != manifest["resources_sha256"]:
            raise ValueError("Snapshot resource checksum mismatch")
        if snapshot_hash != manifest["snapshot_id"]:
            raise ValueError("Snapshot identity mismatch")
        index_path = directory / "resources.sqlite"
        if manifest.get("schema_version", 1) >= 2 and not index_path.is_file():
            raise ValueError("Snapshot index is missing")
        if index_path.exists():
            if on_progress:
                on_progress("Verifying snapshot index checksum")
            if file_digest(index_path) != manifest["index_sha256"]:
                raise ValueError("Snapshot index checksum mismatch")
            instance = cls.__new__(cls)
            instance.id = manifest["snapshot_id"]
            instance.resources = SQLiteResources(index_path)
            instance.patients = [row[0] for row in instance.resources.connection.execute(
                "SELECT key FROM resources WHERE key >= 'Patient/' AND key < 'Patient0' ORDER BY key")]
            if on_progress:
                on_progress(f"Snapshot ready: {len(instance.patients)} patients")
            return instance
        return cls(iter_jsonl(resources_path), manifest["snapshot_id"])


class SQLiteResources(Mapping):
    """Read-only resource mapping backed by SQLite; values are decoded on demand."""

    def __init__(self, path):
        self.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        self.connection.execute("PRAGMA cache_size=-8192")
        self.patient_cache = {}

    def __getitem__(self, key):
        if key in self.patient_cache:
            return self.patient_cache[key]
        row = self.connection.execute("SELECT body FROM resources WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def __contains__(self, key):
        if key in self.patient_cache:
            return True
        return self.connection.execute("SELECT 1 FROM resources WHERE key=?", (key,)).fetchone() is not None

    def __iter__(self):
        for row in self.connection.execute("SELECT key FROM resources ORDER BY key"):
            yield row[0]

    def __len__(self):
        return self.connection.execute("SELECT count(*) FROM resources").fetchone()[0]

    def items(self):
        for key, body in self.connection.execute("SELECT key,body FROM resources ORDER BY key"):
            yield key, json.loads(body)
