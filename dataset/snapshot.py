"""Stream preserved FHIR bundles into a checksummed, disk-backed snapshot."""

import argparse
from contextlib import contextmanager
import os
import json
import re
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from dataset.bundles import iter_bundle_entries
from dataset.common import Snapshot, SQLiteResources, canonical, codes, digest, file_digest, patient_ref, resource_id, sequence_digest, write_json, write_jsonl


def walk_references(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "reference" and isinstance(child, str):
                yield value, child
            else:
                yield from walk_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_references(child)


def without_identity(value):
    if isinstance(value, dict):
        return {k: without_identity(v) for k, v in value.items()
                if k not in {"id", "identifier", "meta", "text", "reference"}}
    if isinstance(value, list):
        return [without_identity(v) for v in value]
    return value


def create_index(directories, connection):
    connection.executescript("""
        PRAGMA cache_size=-8192;
        CREATE TABLE resources (key TEXT PRIMARY KEY, resource_type TEXT, patient TEXT, body TEXT);
        CREATE TABLE aliases (alias TEXT PRIMARY KEY, key TEXT);
        CREATE TABLE identifiers (resource_type TEXT, system TEXT, value TEXT, key TEXT,
                                 PRIMARY KEY(resource_type,system,value,key));
        CREATE INDEX identifiers_value ON identifiers(resource_type,value);
    """)
    source_files, repeated, patient_targets = {}, 0, 0
    for directory in directories:
        paths = sorted(directory.glob("*.json"))
        if not paths:
            raise ValueError(f"No FHIR bundles in {directory}")
        generation_path = directory.parent / "generation.json"
        expected = None
        if generation_path.exists():
            generation = json.loads(generation_path.read_text())
            if generation["status"] != "complete":
                raise ValueError(f"Generation incomplete: {generation_path}")
            expected = generation["files"]
            source_files[str(generation_path.resolve())] = file_digest(generation_path)
            expected_paths = {name for name in expected if name.startswith("fhir/") and name.endswith(".json")}
            actual_paths = {str(p.relative_to(directory.parent)) for p in paths}
            if expected_paths != actual_paths:
                raise ValueError(f"Missing/extra bundles relative to generation manifest in {directory}")
            patient_targets += generation["patient_count"]
        for file_number, path in enumerate(paths, 1):
            actual_hash = file_digest(path)
            if expected is not None and expected.get(str(path.relative_to(directory.parent))) != actual_hash:
                raise ValueError(f"Original bundle changed: {path}")
            source_files[str(path.resolve())] = actual_hash
            for entry in iter_bundle_entries(path):
                resource = entry.get("resource", {})
                if not resource.get("resourceType"):
                    raise ValueError(f"Missing resource in {path}")
                full_url = entry.get("fullUrl", "")
                if not resource.get("id") and full_url.startswith("urn:uuid:"):
                    resource["id"] = full_url.removeprefix("urn:uuid:")
                if not re.fullmatch(r"[A-Za-z0-9.-]{1,64}", str(resource.get("id", ""))):
                    raise ValueError(f"Invalid/missing logical resource ID in {path}")
                key, body = resource_id(resource), canonical(resource)
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO resources VALUES(?,?,?,?)",
                    (key, resource["resourceType"], None, body)).rowcount
                if not inserted:
                    previous = connection.execute("SELECT body FROM resources WHERE key=?", (key,)).fetchone()[0]
                    if previous != body:
                        raise ValueError(f"Conflicting duplicate resource {key} in {path}")
                    repeated += 1
                if full_url:
                    inserted = connection.execute("INSERT OR IGNORE INTO aliases VALUES(?,?)", (full_url, key)).rowcount
                    if not inserted and connection.execute("SELECT key FROM aliases WHERE alias=?", (full_url,)).fetchone()[0] != key:
                        raise ValueError(f"Conflicting fullUrl: {full_url}")
                for identifier in resource.get("identifier", []):
                    if identifier.get("value"):
                        connection.execute("INSERT OR IGNORE INTO identifiers VALUES(?,?,?,?)",
                            (resource["resourceType"], identifier.get("system", ""), identifier["value"], key))
            connection.commit()
            if file_number % 20 == 0 or file_number == len(paths):
                print(f"Indexed {file_number}/{len(paths)} bundles from {directory}", flush=True)
    return source_files, repeated, patient_targets


def normalize_index(connection):
    unresolved_count, examples = 0, []
    observation_codes, component_codes = Counter(), Counter()
    count = 0
    for key, body in connection.execute("SELECT key,body FROM resources ORDER BY key"):
        resource = json.loads(body)
        for container, ref in walk_references(resource):
            if ref.startswith("#"):
                if ref[1:] in {r.get("id") for r in resource.get("contained", [])}:
                    continue
                resolved = None
            else:
                row = connection.execute("SELECT key FROM aliases WHERE alias=?", (ref,)).fetchone()
                resolved = row[0] if row else None
                if not resolved and "?" in ref:
                    parsed = urlparse(ref)
                    rtype = parsed.path.rstrip("/").split("/")[-1]
                    values = parse_qs(parsed.query).get("identifier", [])
                    if len(values) == 1:
                        token = values[0]
                        system, value = token.split("|", 1) if "|" in token else ("", token)
                        if system:
                            rows = connection.execute("SELECT DISTINCT key FROM identifiers WHERE resource_type=? AND system=? AND value=?",
                                                      (rtype, system, value)).fetchmany(2)
                        else:
                            rows = connection.execute("SELECT DISTINCT key FROM identifiers WHERE resource_type=? AND value=?",
                                                      (rtype, value)).fetchmany(2)
                        if len(rows) == 1:
                            resolved = rows[0][0]
                if not resolved:
                    parts = unquote(urlparse(ref).path).rstrip("/").split("/")
                    if "_history" in parts:
                        parts = parts[:parts.index("_history")]
                    candidate = "/".join(parts[-2:])
                    if connection.execute("SELECT 1 FROM resources WHERE key=?", (candidate,)).fetchone():
                        resolved = candidate
            if resolved:
                container["reference"] = resolved
            else:
                unresolved_count += 1
                if len(examples) < 5:
                    examples.append({"source": key, "reference": ref})
        connection.execute("UPDATE resources SET body=?,patient=? WHERE key=?",
                           (canonical(resource), patient_ref(resource), key))
        if resource["resourceType"] == "Observation":
            observation_codes.update(f"{system}|{code}" for system, code in codes(resource.get("code", {})))
            for component in resource.get("component", []):
                component_codes.update(f"{system}|{code}" for system, code in codes(component.get("code", {})))
        count += 1
        if count % 50000 == 0:
            print(f"Normalized {count} resources", flush=True)
    if unresolved_count:
        raise ValueError(f"{unresolved_count} unresolved references; first examples: {examples}")
    print(f"Normalized {count} resources; creating patient index", flush=True)
    connection.execute("CREATE INDEX resources_patient_type ON resources(patient,resource_type)")
    connection.commit()
    return observation_codes, component_codes


def audit_patients(connection, target):
    patients = [r[0] for r in connection.execute("SELECT key FROM resources WHERE resource_type='Patient' ORDER BY key")]
    if not patients:
        raise ValueError("Snapshot has no patients")
    if target and len(patients) != target:
        raise ValueError(f"Expected {target} distinct patients from manifests, found {len(patients)}")
    unknown = connection.execute("""
        SELECT r.key FROM resources r LEFT JOIN resources p ON r.patient=p.key
        WHERE r.patient IS NOT NULL AND (p.key IS NULL OR p.resource_type!='Patient') LIMIT 5
    """).fetchall()
    if unknown:
        raise ValueError(f"Unrecognized patient references: {unknown}")
    duplicates = connection.execute("""
        SELECT system,value FROM identifiers WHERE resource_type='Patient'
        GROUP BY system,value HAVING count(DISTINCT key)>1 LIMIT 5
    """).fetchall()
    if duplicates:
        raise ValueError(f"Duplicate patient identifiers across IDs: {duplicates}")
    histories = defaultdict(list)
    for patient in patients:
        demographic = json.loads(connection.execute("SELECT body FROM resources WHERE key=?", (patient,)).fetchone()[0])
        demographic = {k: demographic.get(k) for k in ("birthDate", "gender", "deceasedDateTime")}
        fingerprints = [digest(without_identity(json.loads(row[0]))) for row in connection.execute(
            "SELECT body FROM resources WHERE patient=?", (patient,))]
        if len(fingerprints) >= 5:
            histories[digest([demographic, sorted(fingerprints)])].append(patient)
    cloned = [group for group in histories.values() if len(group) > 1]
    if cloned:
        raise ValueError(f"Potential cloned clinical histories; inspect before splitting: {cloned[:5]}")
    return patients


def build_snapshot(directories, output):
    if output.exists():
        raise ValueError(f"Snapshot already exists: {output}; choose a new output directory.")
    with tempfile.TemporaryDirectory(prefix="beyond-fhir-index-") as temporary:
        index_path = Path(temporary) / "resources.sqlite"
        connection = sqlite3.connect(index_path)
        try:
            source_files, repeated, target = create_index(directories, connection)
            observation_codes, component_codes = normalize_index(connection)
            print("References normalized; checking patient integrity", flush=True)
            patients = audit_patients(connection, target)
            print("Patient integrity passed; exporting immutable snapshot", flush=True)
            resource_counts = dict(connection.execute("SELECT resource_type,count(*) FROM resources GROUP BY resource_type"))
            # Alias and identifier tables are construction-only; retain a compact read index.
            connection.executescript("DROP TABLE aliases; DROP TABLE identifiers;")
            connection.execute("VACUUM")
            output.mkdir(parents=True)
            def resources():
                for row in connection.execute("SELECT body FROM resources ORDER BY key"):
                    yield json.loads(row[0])
            snapshot_id = sequence_digest(resources())
            write_jsonl(output / "resources.jsonl", resources())
            connection.close()
            shutil.copyfile(index_path, output / "resources.sqlite")
            manifest = {
                "schema_version": 2, "snapshot_id": snapshot_id,
                "resources_sha256": file_digest(output / "resources.jsonl"),
                "index_sha256": file_digest(output / "resources.sqlite"),
                "patient_count": len(patients), "resource_counts": resource_counts,
                "identical_duplicate_entries": repeated, "source_files": source_files,
            }
            write_json(output / "manifest.json", manifest)
            write_json(output / "audit.json", {
                "snapshot_id": snapshot_id, "patient_ids": patients,
                "observation_codes": dict(observation_codes), "component_codes": dict(component_codes),
                "unresolved_references": [], "duplicate_patient_identifiers": [], "potential_cloned_histories": [],
            })
            return manifest
        finally:
            connection.close()



DEFAULT_BUNDLES = Path(__file__).resolve().parents[1] / "data/raw/batch-001/fhir"


def add_source_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bundles", type=Path, nargs="+", help="Original Synthea FHIR directories; defaults to data/raw/batch-001/fhir")
    group.add_argument("--snapshot", type=Path, help="Optional explicitly supplied normalized snapshot")


@contextmanager
def open_records(bundles=None, snapshot=None, on_progress=print):
    """Use a supplied snapshot or construct one temporary SQLite index from raw FHIR.

    No resource export, audit file, or permanent snapshot directory is created.
    """
    if snapshot is not None:
        records = Snapshot.load(snapshot, on_progress=on_progress)
        try:
            yield records
        finally:
            if isinstance(records.resources, SQLiteResources):
                records.resources.connection.close()
        return
    directories = [Path(p) for p in (bundles or [DEFAULT_BUNDLES])]
    with tempfile.TemporaryDirectory(prefix="beyond-fhir-source-", dir=os.environ.get("SLURM_TMPDIR") or None) as temporary:
        index_path = Path(temporary) / "resources.sqlite"
        connection = sqlite3.connect(index_path)
        # This derived index is discarded on every failure; rollback copies add
        # gigabytes of temporary I/O without providing a recoverable artifact.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        # Keep index-build sorting out of SQLite's separate temporary-file path.
        connection.execute("PRAGMA temp_store=MEMORY")
        records = None
        try:
            _, _, target = create_index(directories, connection)
            normalize_index(connection)
            if on_progress:
                on_progress("References normalized; checking patient integrity")
            patients = audit_patients(connection, target)
            if on_progress:
                on_progress("Patient integrity passed; computing source identity")
            identity = sequence_digest(json.loads(r[0]) for r in connection.execute("SELECT body FROM resources ORDER BY key"))
            connection.close()
            records = Snapshot.__new__(Snapshot)
            records.id, records.patients, records.resources = identity, patients, SQLiteResources(index_path)
            if on_progress:
                on_progress(f"Temporary source index ready: {len(patients)} patients")
            yield records
        finally:
            if records is not None:
                records.resources.connection.close()
            connection.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build_snapshot(args.bundles, args.output)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"Snapshot {manifest['snapshot_id'][:12]}: {manifest['patient_count']} patients")


if __name__ == "__main__":
    main()
