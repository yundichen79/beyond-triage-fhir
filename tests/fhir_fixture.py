"""Filter-sensitive fixture transport, not a substitute for live HAPI tests.

Only implemented fixture semantics are accepted. Unsupported parameters raise
instead of silently producing passing tests with unfiltered patient records.
"""

import copy
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse


def token_match(concept, query):
    if isinstance(concept, str):
        return concept == query
    if not isinstance(concept, dict):
        return False
    coding = concept.get("coding", [concept] if "code" in concept else [])
    if "|" in query:
        system, code = query.split("|", 1)
        return any(c.get("code") == code and (not system or c.get("system") == system) for c in coding)
    return any(c.get("code") == query for c in coding)


def date_value(resource, key):
    if key == "date":
        return (resource.get("period", {}).get("start") or resource.get("effectiveDateTime")
                or resource.get("effectivePeriod", {}).get("start") or resource.get("performedDateTime")
                or resource.get("performedPeriod", {}).get("start"))
    return resource.get({"authoredon": "authoredOn", "onset-date": "onsetDateTime",
                         "abatement-date": "abatementDateTime"}[key])


def date_match(value, query):
    if not value:
        return False
    prefix = query[:2] if query[:2] in {"eq", "ne", "lt", "le", "gt", "ge"} else "eq"
    text = query[2:] if query[:2] == prefix else query
    actual = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if actual.tzinfo is None:
        actual = actual.replace(tzinfo=timezone.utc)
    boundary = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if boundary.tzinfo is None:
        boundary = boundary.replace(tzinfo=actual.tzinfo)
    return {"eq": actual == boundary, "ne": actual != boundary, "lt": actual < boundary,
            "le": actual <= boundary, "gt": actual > boundary, "ge": actual >= boundary}[prefix]


def matches(resource, key, value):
    if key in {"date", "authoredon", "onset-date", "abatement-date"}:
        return date_match(date_value(resource, key), value)
    if key in {"_id", "status", "intent"}:
        return resource.get("id" if key == "_id" else key) == value
    if key == "encounter":
        return resource.get("encounter", {}).get("reference") == value
    if key in {"class", "code", "type", "category", "clinical-status", "verification-status"}:
        concept = resource.get({"clinical-status": "clinicalStatus",
                                "verification-status": "verificationStatus"}.get(key, key), {})
        concepts = concept if isinstance(concept, list) else [concept]
        return any(token_match(c, value) for c in concepts)
    if key in {"component-code", "combo-code"}:
        concepts = [c.get("code", {}) for c in resource.get("component", [])]
        if key == "combo-code":
            concepts.append(resource.get("code", {}))
        return any(token_match(c, value) for c in concepts)
    raise AssertionError(f"Fixture transport does not implement filter {key}")


def search(snapshot, path):
    parsed = urlparse(path)
    kind = parsed.path.rstrip("/").split("/")[-1]
    params = parse_qs(parsed.query, keep_blank_values=True)
    supported = {"patient", "_count", "_total", "_sort", "_id", "status", "intent",
                 "encounter", "class", "code", "type", "category", "clinical-status",
                 "verification-status", "component-code", "combo-code", "date",
                 "authoredon", "onset-date", "abatement-date"}
    if set(params) - supported:
        raise AssertionError(f"Fixture transport does not implement filters {sorted(set(params) - supported)}")
    resources = list(snapshot.for_patient(params["patient"][0], kind))
    for key, constraints in params.items():
        if key in {"patient", "_count", "_total", "_sort"}:
            continue
        # Repeated parameters are AND; comma-separated values are OR.
        for constraint in constraints:
            resources = [r for r in resources if any(matches(r, key, v) for v in constraint.split(","))]
    for sort in reversed(params.get("_sort", [""])[0].split(",")):
        if not sort:
            continue
        key = sort.lstrip("-")
        if key == "_id":
            resources.sort(key=lambda r: r["id"], reverse=sort.startswith("-"))
        elif key in {"date", "authoredon", "onset-date", "abatement-date"}:
            resources.sort(key=lambda r: date_value(r, key) or "", reverse=sort.startswith("-"))
        else:
            raise AssertionError(f"Fixture transport does not implement sort {sort}")
    count = int(params.get("_count", [len(resources)])[0])
    if len(resources) > count:
        raise AssertionError("Fixture search needs pagination: use explicit paginated response fixtures")
    return {"resourceType": "Bundle", "total": len(resources),
            "entry": [{"resource": copy.deepcopy(r)} for r in resources]}
