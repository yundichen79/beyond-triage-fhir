"""Patient-scoped, paginated FHIR tools with explicit completeness and errors."""

import re
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


def function(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}}


TOOL_DEFINITIONS = [
    function("search_fhir", "Search the current patient's FHIR records. Returns one page, total, cumulative received count, complete, and an opaque next_page cursor. Follow next_page when more matches are needed. Clinical projections omit narratives and server metadata. Dates support repeated bounds such as [geTIMESTAMP, ltTIMESTAMP]. Supported parameters: "
             + "; ".join(kind + ": " + ", ".join(sorted(fields)) for kind, fields in SEARCH_FIELDS.items()),
             {"resource_type": {"type": "string", "enum": sorted(SEARCH_FIELDS)},
              "params": {"type": "object", "description": "FHIR parameters, with string or string-array values. Patient scope is supplied automatically."},
              "count": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}}, ["resource_type"]),
    function("read_fhir", "Read a resource by a reference already seen in tool results, such as Observation/id or Medication/id. The supplied patient reference is also readable. Contained # references are available inside their parent resource.",
             {"reference": {"type": "string"}}, ["reference"]),
    function("next_page", "Retrieve the next page of an earlier search using its opaque cursor. No records are silently truncated. A cursor can be consumed once.",
             {"cursor": {"type": "string"}}, ["cursor"]),
    function("finish", "Submit the structured answer using the public answer contract, and cite retrieved ResourceType/id references. Use finish alone, after reading the evidence; it terminates the episode.",
             {"answer": {"type": "object"}, "evidence": {"type": "array", "items": {"type": "string"}}}, ["answer", "evidence"]),
]


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
            if name == "search_fhir":
                if set(arguments) - {"resource_type", "params", "count"}:
                    raise ValueError("Unexpected search arguments")
                kind = arguments.get("resource_type")
                if kind not in SEARCH_FIELDS:
                    raise ValueError("Unsupported searchable resource type")
                params, count = arguments.get("params", {}), arguments.get("count", 10)
                if not isinstance(params, dict) or set(params) - SEARCH_FIELDS[kind]:
                    raise ValueError("Unsupported FHIR search parameters; patient scope is automatic")
                if type(count) is not int or not 1 <= count <= 50:
                    raise ValueError("count must be an integer between 1 and 50")
                for value in params.values():
                    if not (isinstance(value, str) or isinstance(value, list) and value and all(isinstance(v, str) for v in value)):
                        raise ValueError("Search values must be strings or nonempty string arrays")
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
