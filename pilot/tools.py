"""Patient-scoped, paginated FHIR tools with explicit completeness and errors."""

import re
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from copy import deepcopy
from urllib.parse import urlencode, urljoin

from dataset.common import patient_ref, resource_id


SEARCH_FIELDS = {
    "Encounter": {"date", "status", "class", "type", "_id", "_sort"},
    "Condition": {"code", "clinical-status", "verification-status", "onset-date", "abatement-date", "encounter", "_id", "_sort"},
    "Observation": {"code", "component-code", "combo-code", "date", "status", "encounter", "category", "_id", "_sort"},
    "DiagnosticReport": {"code", "date", "status", "encounter", "_id", "_sort"},
    "MedicationRequest": {"code", "authoredon", "status", "intent", "encounter", "medication", "_id", "_sort"},
    "Procedure": {"code", "date", "status", "encounter", "_id", "_sort"},
}
# Supported values/fields for this pinned R4 interface. A valid parameter name
# alone must not turn a semantically invalid query into apparent absence evidence.
STATUSES = {
    "Encounter": {"planned", "arrived", "triaged", "in-progress", "onleave", "finished", "cancelled", "entered-in-error", "unknown"},
    "Procedure": {"preparation", "in-progress", "not-done", "on-hold", "stopped", "completed", "entered-in-error", "unknown"},
    "MedicationRequest": {"active", "on-hold", "cancelled", "completed", "entered-in-error", "stopped", "draft", "unknown"},
    "Observation": {"registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"},
    "DiagnosticReport": {"registered", "partial", "preliminary", "final", "amended", "corrected", "appended", "cancelled", "entered-in-error", "unknown"},
}
CONDITION_STATUSES = {
    "clinical-status": {"active", "recurrence", "relapse", "inactive", "remission", "resolved"},
    "verification-status": {"unconfirmed", "provisional", "differential", "confirmed", "refuted", "entered-in-error"},
}
SORT_FIELDS = {
    kind: {"_id", "_lastUpdated", date_field} for kind, date_field in {
        "Encounter": "date", "Condition": "onset-date", "Observation": "date",
        "DiagnosticReport": "date", "MedicationRequest": "authoredon", "Procedure": "date",
    }.items()
}
SORT_FIELDS["Condition"].update({"abatement-date", "recorded-date"})
FIELDS = {
    "Patient": {"birthDate", "gender", "deceasedDateTime"},
    "Encounter": {"status", "class", "subject", "period", "diagnosis", "reasonReference"},
    "Condition": {"code", "clinicalStatus", "verificationStatus", "subject", "encounter", "onsetDateTime", "abatementDateTime", "recordedDate"},
    "Observation": {"status", "code", "subject", "encounter", "effectiveDateTime", "effectivePeriod", "valueQuantity", "component", "dataAbsentReason"},
    "DiagnosticReport": {"status", "code", "subject", "encounter", "effectiveDateTime", "effectivePeriod", "issued", "result"},
    "MedicationRequest": {"status", "intent", "subject", "encounter", "medicationCodeableConcept", "medicationReference", "authoredOn", "reasonReference"},
    "Medication": {"code", "status", "ingredient"},
    "Procedure": {"status", "code", "subject", "encounter", "performedDateTime", "performedPeriod", "reasonReference"},
}
REFERENCE = re.compile(r"^([A-Za-z]+)/([A-Za-z0-9.-]{1,64})$")
SEARCH_GUIDANCE = """Use FHIR search syntax, not resource JSON fields.
Patient scope is automatic; omit patient and subject. Put count (1..50) at the
top level, outside params. _sort uses the resource's field (prefix - for newest):
date for Encounter/Observation/DiagnosticReport/Procedure, onset-date for Condition,
authoredon for MedicationRequest. Never use sort, date:asc, or date desc.
Encounter statuses include finished; Procedure statuses include completed;
MedicationRequest statuses include active/completed. final/amended/corrected are
Observation and DiagnosticReport statuses, not encounter/order/procedure statuses.
Condition uses clinical-status (e.g. active/resolved) and verification-status
(e.g. confirmed/refuted), not status or observation status values.
Arrays mean repeated AND filters:
date=["ge2025-01-01", "lt2026-01-01"]. Alternatives use a comma-separated string:
status="final,amended,corrected". Token codes are strings such as "system|code",
not coding objects. A date prefix and value form ONE string, e.g. "lt2026-01-01".
Encounter inpatient/ambulatory class is class="IMP"/"AMB", not type="inpatient".
Encounter references are "Encounter/logical-id", not "ambulatory" or a patient.
Only use encounter references already seen in tool results; search encounters first.
Condition dates use onset-date/abatement-date; MedicationRequest dates use authoredon.
reasonReference is a returned resource field, not a supported search parameter:
retrieve orders/procedures, inspect their reasonReference, then read those targets.
Search CBC reports using DiagnosticReport code="http://loinc.org|58410-2";
read their result references for observations. Blood pressure may be in Observation
components (component-code or combo-code); category="blood-pressure" is not its code.
Do not invent lower date bounds. Empty filtered results do not prove chart-wide
absence: check filters and broaden the search within the question's scope.
"""
PARAMETER_SCHEMA = {"anyOf": [
    {"type": "string"},
    {"type": "array", "items": {"type": "string"}, "minItems": 1},
]}


def function(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}}


TOOL_DEFINITIONS = [
    function("search_fhir", "Search the current patient's FHIR records. Returns one page, total, cumulative received count, complete, and an opaque next_page cursor. Follow next_page when more matches are needed. Clinical projections omit narratives and server metadata. Dates support repeated bounds such as [geTIMESTAMP, ltTIMESTAMP]. Supported parameters: "
             + "; ".join(kind + ": " + ", ".join(sorted(fields)) for kind, fields in SEARCH_FIELDS.items())
             + ". Supported status values: "
             + "; ".join(kind + ": " + ", ".join(sorted(values)) for kind, values in STATUSES.items())
             + "; Condition: " + "; ".join(key + ": " + ", ".join(sorted(values)) for key, values in CONDITION_STATUSES.items())
             + ". Supported sort fields: "
             + "; ".join(kind + ": " + ", ".join(sorted(values)) for kind, values in SORT_FIELDS.items())
             + ". " + SEARCH_GUIDANCE,
             {"resource_type": {"type": "string", "enum": sorted(SEARCH_FIELDS)},
              "params": {"type": "object", "description": "Only parameters listed for resource_type; arrays are AND, comma-separated strings are OR.",
                         "properties": {key: PARAMETER_SCHEMA for key in sorted(set().union(*SEARCH_FIELDS.values()))},
                         "additionalProperties": False},
              "count": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}}, ["resource_type"]),
    function("read_fhir", "Read a resource by a reference already seen in tool results, such as Observation/id or Medication/id. The supplied patient reference is also readable. Contained # references are available inside their parent resource.",
             {"reference": {"type": "string"}}, ["reference"]),
    function("next_page", "Retrieve the next page of an earlier search using its opaque cursor. No records are silently truncated. A cursor can be consumed once.",
             {"cursor": {"type": "string"}}, ["cursor"]),
    function("calculate", "Compute later minus earlier using decimal arithmetic and half-even rounding. Supply numeric strings copied from retrieved measurements; returns a string rounded to the requested decimal places. This tool does not retrieve or verify evidence.",
             {"earlier": {"type": "string"}, "later": {"type": "string"},
              "decimal_places": {"type": "integer", "minimum": 0, "maximum": 6}},
             ["earlier", "later", "decimal_places"]),
    function("finish", "Submit the structured answer using the public answer contract, and cite retrieved ResourceType/id references. Use finish alone, after reading the evidence; it terminates the episode.",
             {"answer": {"type": "object"}, "evidence": {"type": "array", "items": {"type": "string"}}}, ["answer", "evidence"]),
]


def tools_for_answer(answer_schema):
    """Bind finish to a question's public schema; never mutate shared definitions."""
    definitions = deepcopy(TOOL_DEFINITIONS)
    definitions[-1]["function"]["parameters"]["properties"]["answer"] = deepcopy(answer_schema)
    return definitions


def projection(resource):
    fields = FIELDS[resource["resourceType"]] | {"resourceType", "id"}
    result = {k: v for k, v in resource.items() if k in fields}
    if resource.get("contained"):
        result["contained"] = [projection(r) for r in resource["contained"] if r.get("resourceType") in FIELDS]
    return result


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get("reference"), str):
            yield value["reference"]
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


class PatientTools:
    def __init__(self, client, patient):
        if not REFERENCE.fullmatch(patient) or not patient.startswith("Patient/"):
            raise ValueError("Expected a Patient/logical-id reference")
        self.client, self.patient = client, patient
        self.allowed = {patient}
        self.seen = set()
        self.cursors, self.searches = {}, []
        self.calls, self.http_requests, self.errors = 0, 0, 0

    def validate_resource(self, resource, expected_type):
        if resource.get("resourceType") != expected_type or not resource.get("id"):
            raise ValueError("FHIR returned a missing or incorrectly typed resource")
        owner = patient_ref(resource)
        if (expected_type == "Patient" and resource_id(resource) != self.patient
                or expected_type in SEARCH_FIELDS and owner != self.patient
                or owner and owner != self.patient):
            raise ValueError("FHIR returned a resource outside the current patient")

    def observe(self, resource, expected_type):
        self.validate_resource(resource, expected_type)
        result = projection(resource)
        self.seen.add(resource_id(resource))
        self.allowed.add(resource_id(resource))
        for ref in references(result):
            match = REFERENCE.fullmatch(ref)
            if match and match[1] in FIELDS and (match[1] != "Patient" or ref == self.patient):
                self.allowed.add(ref)
        return result

    def page(self, state, url):
        # The shared FHIR client also rejects pagination URLs outside its server.
        absolute = urljoin(self.client.base + "/", url)
        if absolute in state["urls"]:
            raise ValueError("FHIR pagination cycle")
        self.http_requests += 1
        bundle = self.client.request(url)
        if bundle.get("resourceType") != "Bundle":
            raise ValueError("FHIR search did not return a Bundle")
        total = bundle.get("total")
        if total is not None:
            if not isinstance(total, int) or isinstance(total, bool) or total < 0:
                raise ValueError("FHIR returned an invalid search total")
            if state["total"] is not None and state["total"] != total:
                raise ValueError("FHIR search total changed during pagination")
            state["total"] = total
        resources, ids = [], set(state["ids"])
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if entry.get("search", {}).get("mode") == "outcome" or resource.get("resourceType") == "OperationOutcome":
                raise ValueError("FHIR search returned a warning or error")
            if resource.get("resourceType") != state["type"]:
                raise ValueError("FHIR search returned an unexpected resource type")
            self.validate_resource(resource, state["type"])
            key = resource_id(resource)
            if key not in ids:
                resources.append(resource)
                ids.add(key)
        following = next((link["url"] for link in bundle.get("link", []) if link.get("relation") == "next"), None)
        if state["total"] is not None and (len(ids) > state["total"]
                or not following and len(ids) != state["total"]):
            raise ValueError("FHIR pagination ended with an incomplete or inconsistent result count")
        cursor = None
        if following:
            following = urljoin(absolute, following)
            if following == absolute or following in state["urls"]:
                raise ValueError("FHIR pagination cycle")
            cursor = f"page_{len(self.cursors) + 1}"
            self.cursors[cursor] = (state, following)
        state["urls"].add(absolute)
        state["ids"], state["complete"] = ids, following is None
        return {"resources": [self.observe(r, state["type"]) for r in resources], "total": state["total"], "received": len(ids),
                "complete": state["complete"], "next_page": cursor}

    def execute(self, name, arguments):
        self.calls += 1
        try:
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            if name == "calculate":
                if set(arguments) != {"earlier", "later", "decimal_places"}:
                    raise ValueError("calculate requires earlier, later, and decimal_places")
                places = arguments["decimal_places"]
                if type(places) is not int or not 0 <= places <= 6:
                    raise ValueError("decimal_places must be an integer from 0 to 6")
                values = [arguments[key] for key in ("earlier", "later")]
                if any(not isinstance(v, str) or not re.fullmatch(r"-?\d{1,18}(?:\.\d{1,18})?", v) for v in values):
                    raise ValueError("Use finite decimal strings with at most 18 integer and 18 fractional digits")
                with localcontext() as context:
                    context.prec = 50
                    earlier, later = map(Decimal, values)
                    result = (later - earlier).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)
                return {"value": format(result, "f"), "operation": "later_minus_earlier",
                        "decimal_places": places}
            if name == "search_fhir":
                if set(arguments) - {"resource_type", "params", "count"}:
                    raise ValueError("Unexpected search arguments")
                kind = arguments.get("resource_type")
                if kind not in SEARCH_FIELDS:
                    raise ValueError("Unsupported searchable resource type")
                params, count = arguments.get("params", {}), arguments.get("count", 10)
                if not isinstance(params, dict):
                    raise ValueError("params must be an object of FHIR parameter names and string values")
                unsupported = sorted(set(params) - SEARCH_FIELDS[kind])
                if unsupported:
                    raise ValueError(f"Unsupported {kind} parameters {unsupported}. Allowed: "
                        + ", ".join(sorted(SEARCH_FIELDS[kind]))
                        + ". Patient scope is automatic; count belongs outside params. "
                        "Use _sort, not sort; reasonReference is inspected on retrieved resources, not searched.")
                if type(count) is not int or not 1 <= count <= 50:
                    raise ValueError("count must be an integer between 1 and 50")
                for key, value in params.items():
                    if not (isinstance(value, str) or isinstance(value, list) and value and all(isinstance(v, str) for v in value)):
                        raise ValueError(f"params.{key} must be a string or nonempty string array. "
                            "Use system|code for tokens, not coding objects. Arrays are repeated AND filters; "
                            "use a comma-separated string for alternatives.")
                    values = value if isinstance(value, list) else [value]
                    if key in {"class", "encounter"} and len(values) > 1:
                        groups = [set(v.split(",")) for v in values]
                        if not set.intersection(*groups):
                            raise ValueError(f"Contradictory {key} filters: arrays mean AND. "
                                             "Use one comma-separated string for alternatives, or query one at a time.")
                    if key == "_sort" and any(not re.fullmatch(r"-?[A-Za-z_][A-Za-z0-9_-]*(,-?[A-Za-z_][A-Za-z0-9_-]*)*", v) for v in values):
                        raise ValueError("params._sort uses field or -field, e.g. date or -date; no :asc or spaces")
                    if key == "_sort":
                        invalid = sorted({field.lstrip("-") for v in values for field in v.split(",")} - SORT_FIELDS[kind])
                        if invalid:
                            raise ValueError(f"Unsupported {kind} sort fields {invalid}. Supported: "
                                             + ", ".join(sorted(SORT_FIELDS[kind])))
                    if key == "status" or kind == "Condition" and key in CONDITION_STATUSES:
                        allowed = STATUSES[kind] if key == "status" else CONDITION_STATUSES[key]
                        groups = [{token.rsplit("|", 1)[-1] for token in v.split(",")} for v in values]
                        invalid = sorted(set().union(*groups) - allowed)
                        if invalid:
                            raise ValueError(f"Invalid {kind} {key} values {invalid}. Allowed: "
                                + ", ".join(sorted(allowed))
                                + ". Observation statuses do not apply to encounters, orders, or procedures.")
                        if not set.intersection(*groups):
                            raise ValueError("Contradictory status filters: arrays mean AND. "
                                             "Use one comma-separated string for status alternatives.")
                    if key == "encounter":
                        refs = [ref for v in values for ref in v.split(",")]
                        if any(not REFERENCE.fullmatch(ref) or not ref.startswith("Encounter/") for ref in refs):
                            raise ValueError("params.encounter must contain Encounter/logical-id references, not a class or patient")
                        if any(ref not in self.allowed for ref in refs):
                            raise ValueError("Unseen encounter reference. Search encounters first and use "
                                             "Encounter/logical-id references actually returned in tool results.")
                    if key in {"date", "onset-date", "abatement-date", "authoredon"} and any(
                            not re.fullmatch(r"(?:(?:eq|ne|gt|lt|ge|le|sa|eb|ap))?\d{4}(?:-\d{2}(?:-\d{2}(?:T[^\s,\[\]]+)?)?)?", v) for v in values):
                        raise ValueError(f"params.{key} needs one date per string, e.g. "
                            '["ge2025-01-01", "lt2026-01-01"]; do not split prefixes or stringify arrays')
                params = dict(params, patient=self.patient, _count=count, _total="accurate")
                state = {"type": kind, "total": None, "ids": set(), "urls": set(), "complete": False}
                self.searches.append(state)
                return self.page(state, kind + "?" + urlencode(params, doseq=True))
            if name == "next_page":
                if set(arguments) != {"cursor"} or not isinstance(arguments["cursor"], str) or not self.cursors.get(arguments["cursor"]):
                    raise ValueError("Unknown pagination cursor")
                state, url = self.cursors[arguments["cursor"]]
                result = self.page(state, url)
                # Keep the consumed key so future cursors always have unique IDs.
                self.cursors[arguments["cursor"]] = None
                return result
            if name == "read_fhir":
                ref = arguments.get("reference")
                if set(arguments) != {"reference"} or not isinstance(ref, str) or ref not in self.allowed:
                    raise ValueError("Read a resource reference previously returned by a tool")
                self.http_requests += 1
                resource = self.client.request(ref)
                if resource_id(resource) != ref:
                    raise ValueError("FHIR read returned a different resource ID")
                return {"resource": self.observe(resource, ref.split("/", 1)[0])}
            raise ValueError("Unknown tool name")
        except InterruptedError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.errors += 1
            return {"error": str(exc), "complete": False}
