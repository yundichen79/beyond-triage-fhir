"""Select complete patient charts and their transitive reference closure for import."""

from collections import Counter
from collections.abc import Mapping
import json

from dataset.common import SQLiteResources, digest, patient_ref
from dataset.snapshot import walk_references


class SelectedResources(Mapping):
    def __init__(self, parent, keys, patients):
        self.parent, self.keys = parent, frozenset(keys)
        self.patients = sorted(patients)

    def __getitem__(self, key):
        if key not in self.keys:
            raise KeyError(key)
        return self.parent[key]

    def __iter__(self):
        return iter(sorted(self.keys))

    def __len__(self):
        return len(self.keys)

    def items(self):
        emitted = set()
        if isinstance(self.parent, SQLiteResources):
            for patient in self.patients:
                yield patient, self.parent[patient]
                emitted.add(patient)
                for key, body in self.parent.connection.execute(
                        "SELECT key,body FROM resources WHERE patient=? ORDER BY rowid", (patient,)):
                    if key in self.keys and key not in emitted:
                        emitted.add(key)
                        yield key, json.loads(body)
        for key in sorted(self.keys - emitted):
            yield key, self.parent[key]


class PatientScope:
    def __init__(self, parent, patients, keys):
        self.id, self.parent = parent.id, parent
        self.patients = sorted(patients)
        self.resources = SelectedResources(parent.resources, keys, patients)
        self.scope = {
            "source_snapshot_id": self.id,
            "patients": self.patients,
            "resource_count": len(keys),
            "resource_key_digest": digest(sorted(keys)),
            "resource_counts": dict(sorted(Counter(k.split('/')[0] for k in keys).items())),
            "selection": "All directly patient-owned resources plus transitive reference closure",
        }

    def for_patient(self, patient, resource_type):
        if patient not in self.patients:
            raise KeyError(patient)
        return self.parent.for_patient(patient, resource_type)


def scope_for_patients(snapshot, patients):
    patients = set(patients)
    if not patients or not patients.issubset(snapshot.patients):
        raise ValueError("Import scope must contain known snapshot patients")
    keys = set(patients)
    if isinstance(snapshot.resources, SQLiteResources):
        for patient in sorted(patients):
            keys.update(row[0] for row in snapshot.resources.connection.execute(
                "SELECT key FROM resources WHERE patient=?", (patient,)))
    else:
        keys.update(key for key, resource in snapshot.resources.items()
                    if patient_ref(resource) in patients)
    queue = []

    def visit(key, resource):
        owner = patient_ref(resource)
        if ((key.startswith("Patient/") and key not in patients)
                or (owner and owner.startswith("Patient/") and owner not in patients)):
            raise ValueError(f"Reference closure crosses into an unselected patient: {key}")
        for _, ref in walk_references(resource):
            if ref.startswith("#"):
                continue
            if ref not in keys:
                if ref not in snapshot.resources:
                    raise ValueError(f"Unresolved import reference: {key} -> {ref}")
                keys.add(ref)
                queue.append(ref)

    # Stream chart bodies in storage order. Random reads of tens of thousands of
    # SQLite values are expensive on a shared filesystem and aren't needed here.
    if isinstance(snapshot.resources, SQLiteResources):
        for patient in sorted(patients):
            visit(patient, snapshot.resources[patient])
            for key, body in snapshot.resources.connection.execute(
                    "SELECT key,body FROM resources WHERE patient=? ORDER BY rowid", (patient,)):
                visit(key, json.loads(body))
    else:
        for key in sorted(keys):
            visit(key, snapshot.resources[key])
    while queue:
        key = queue.pop()
        visit(key, snapshot.resources[key])
    return PatientScope(snapshot, patients, keys)
