"""Value-free public answer shapes; validation never receives evaluator labels.

The visible question selects one public shape. No evaluator metadata is used.
Validation checks structure and types, not clinical correctness or chart support.
"""

from datetime import datetime
from decimal import Decimal, InvalidOperation
import re


INTERFACE_VERSION = "5"
CONCEPT = {"system": "string", "code": "string", "name": "string"}
CONDITION = {"condition": CONCEPT, "onset": ("date", None)}
FOUND = {"status": {"found"}, "time": "timestamp", "value": "number", "unit": "string"}
MISSING = {"status": {"missing_value"}, "time": "timestamp", "value": None, "unit": "string"}
ABSENT = {"status": {"no_matching_record"}}
MEASUREMENT = (FOUND, MISSING, ABSENT)
RECORDED_MEASUREMENT = (FOUND, MISSING)
CBC = {"report_time": "timestamp", "hemoglobin": MEASUREMENT, "platelets": MEASUREMENT}
PRODUCTS = {"both": [CONCEPT], "only_first": [CONCEPT], "only_second": [CONCEPT]}
STAY = {"admitted_at": "timestamp", "discharged_at": "timestamp"}
FOLLOWUP = {"started_at": "timestamp", "days_after_discharge": "number"}
CHANGE = ("number", None)
ANSWER_SHAPES = {
    "Stay conditions": dict(STAY, conditions=[CONDITION]),
    "Stay orders": dict(STAY, orders=[{
        "medication": CONCEPT, "ordered_at": "timestamp", "recorded_status": "string"}]),
    "Preadmission prescribing history": dict(STAY, conditions_with_prior_orders=[{
        "condition": CONCEPT, "onset": "timestamp", "abatement": ("timestamp", None),
        "orders": [{"medication": CONCEPT, "ordered_at": "timestamp"}]}]),
    "Latest CBC": CBC,
    "CBC comparison": {"previous": CBC, "latest": CBC,
        "hemoglobin_change_g_dL": CHANGE, "platelet_change_10_3_uL": CHANGE},
    "Ambulatory blood pressure": {"visit_started_at": "timestamp",
        "systolic": MEASUREMENT, "diastolic": MEASUREMENT},
    "Discharge follow-up": dict(STAY, followup=(None, dict(FOLLOWUP, conditions=[CONDITION]))),
    "Products across two stays": dict(STAY, previous_admitted_at="timestamp", products=PRODUCTS),
    "Products at discharge follow-up": (
        dict(STAY, followup=None), dict(STAY, followup=FOLLOWUP, products=PRODUCTS)),
    "Procedure indications": {"encounter_started_at": "timestamp", "procedures": [{
        "procedure": CONCEPT, "performed_at": "timestamp", "indications": [CONCEPT]}]},
    "Prescribing indications": {"encounter_started_at": "timestamp", "orders": [{
        "medication": CONCEPT, "indications": [CONCEPT]}]},
    "Glycemic follow-up": (
        {"condition": CONCEPT, "onset": "timestamp", "followup": ABSENT},
        {"condition": CONCEPT, "onset": "timestamp", "followup": RECORDED_MEASUREMENT,
         "baseline": MEASUREMENT, "change_percentage_points": CHANGE}),
    "Renal laboratory follow-up": {"baseline": RECORDED_MEASUREMENT,
        "followup": MEASUREMENT, "change_mg_dL": CHANGE},
    "Subsequent admission": dict(STAY, subsequent_admission=(None, {
        "admitted_at": "timestamp", "days_after_discharge": "number",
        "conditions": [CONDITION], "shared_conditions": [CONCEPT]})),
}

# These are public question prefixes, not dataset family IDs or answer labels.
# Unknown wording fails explicitly rather than silently choosing another task.
QUESTION_PREFIXES = {
    "Stay conditions": "For the most recent completed inpatient stay, list its linked condition codes",
    "Stay orders": "For the most recent completed inpatient stay with at least one recorded non-draft, non-error medication order",
    "Preadmission prescribing history": "Review treatment history before the most recent completed inpatient stay.",
    "Latest CBC": "Review the latest final, amended, or corrected CBC report",
    "CBC comparison": "Review a short-interval CBC trend.",
    "Ambulatory blood pressure": "For the most recent completed ambulatory encounter (class AMB)",
    "Discharge follow-up": "For the most recent completed inpatient stay with a full ",
    "Products across two stays": "Compare coded medication products ordered during the two most recent completed inpatient stays.",
    "Products at discharge follow-up": "For the most recent completed inpatient stay that contains at least one recorded non-draft, non-error medication order",
    "Procedure indications": "Find the most recent completed encounter containing completed procedures",
    "Prescribing indications": "Find the most recent completed encounter containing medication orders",
    "Glycemic follow-up": "Review documented first-year glycemic testing",
    "Renal laboratory follow-up": "Review documented first-year renal laboratory follow-up",
    "Subsequent admission": "After the first recorded completed inpatient stay,",
}


def contract_for_question(question):
    """Route supported benchmark wording using only text shown to the model."""
    if not isinstance(question, str):
        raise ValueError("Question must be text")
    body = re.sub(r"^Patient [A-Za-z0-9.-]+\.\s*", "", question.strip(), count=1)
    matches = [name for name, prefix in QUESTION_PREFIXES.items()
               if body.casefold().startswith(prefix.casefold())]
    if len(matches) != 1:
        raise ValueError("Unsupported or ambiguous public question wording; no answer contract selected")
    name = matches[0]
    return name, ANSWER_SHAPES[name]


def json_schema(shape):
    """Compile the same value-free contract used by the runtime validator."""
    if isinstance(shape, dict):
        return {"type": "object", "properties": {k: json_schema(v) for k, v in shape.items()},
                "required": list(shape), "additionalProperties": False}
    if isinstance(shape, tuple):
        return {"anyOf": [json_schema(v) for v in shape]}
    if isinstance(shape, list):
        return {"type": "array", "items": json_schema(shape[0])}
    if isinstance(shape, set):
        return {"type": "string", "enum": sorted(shape)}
    if shape is None:
        return {"type": "null"}
    if shape == "number":
        return {"anyOf": [{"type": "number"}, {"type": "string",
            "pattern": r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"}]}
    if shape == "timestamp":
        return {"type": "string", "format": "date-time"}
    if shape == "date":
        return {"anyOf": [{"type": "string", "format": "date"},
                          {"type": "string", "format": "date-time"}]}
    return {"type": "string", "minLength": 1}


def shape_error(value, shape, path="answer"):
    if isinstance(shape, tuple):
        errors = [shape_error(value, choice, path) for choice in shape]
        if None in errors:
            return None
        # Prefer feedback for alternatives with the closest object keys.
        candidates = []
        for choice, error in zip(shape, errors):
            if isinstance(value, dict) and isinstance(choice, dict):
                status = choice.get("status")
                mismatch = isinstance(status, set) and (
                    not isinstance(value.get("status"), str) or value["status"] not in status)
                candidates.append((mismatch, len(set(value) ^ set(choice)), error))
        return min(candidates, key=lambda item: item[:2])[2] if candidates else errors[0]
    if isinstance(shape, dict):
        if not isinstance(value, dict):
            return f"{path} must be an object"
        missing, extra = sorted(shape.keys() - value.keys()), sorted(value.keys() - shape.keys())
        if missing or extra:
            return f"{path}: missing fields {missing}; unexpected fields {extra}"
        for key, child in shape.items():
            error = shape_error(value[key], child, f"{path}.{key}")
            if error:
                return error
        return None
    if isinstance(shape, list):
        if not isinstance(value, list):
            return f"{path} must be an array"
        for index, item in enumerate(value):
            error = shape_error(item, shape[0], f"{path}[{index}]")
            if error:
                return error
        return None
    if isinstance(shape, set):
        return None if isinstance(value, str) and value in shape else f"{path} must be one of {sorted(shape)}"
    valid = False
    if shape is None:
        valid = value is None
    elif shape == "string":
        valid = isinstance(value, str) and bool(value)
    elif shape == "number":
        if not isinstance(value, bool) and isinstance(value, (str, int, float)):
            try:
                valid = Decimal(str(value)).is_finite()
            except InvalidOperation:
                pass
    elif shape in {"timestamp", "date"} and isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            valid = parsed.tzinfo is not None or (shape == "date" and len(value) == 10)
        except ValueError:
            pass
    return None if valid else f"{path} must be {shape or 'null'}"


def answer_error(answer, shape=None):
    """With a bound shape, reject other tasks. Unbound mode is for offline audits."""
    if not isinstance(answer, dict):
        return "Answer schema error: answer must be an object"
    if shape is not None:
        error = shape_error(answer, shape)
        if error is None:
            return None
        return (f"Answer schema error: {error}. Follow this question's answer JSON Schema "
                "in the user message and finish tool; do not switch tasks or add a wrapper. "
                "This checks format, not clinical correctness.")
    exact_errors = []
    compatible = set()
    for shape in ANSWER_SHAPES.values():
        error = shape_error(answer, shape)
        if error is None:
            return None
        choices = shape if isinstance(shape, tuple) else (shape,)
        if any(answer.keys() == choice.keys() for choice in choices):
            exact_errors.append(error)
        for choice in choices:
            if answer and answer.keys() < choice.keys():
                compatible.add(tuple(sorted(choice.keys() - answer.keys())))
    if exact_errors:
        detail = "; or ".join(dict.fromkeys(exact_errors))
    elif len(compatible) == 1:
        detail = f"answer is missing fields {list(next(iter(compatible)))}"
    else:
        detail = (f"Top-level fields {sorted(answer)} do not match a complete public structure. "
                  "An incomplete or wrapped object does not identify which task you are answering; "
                  "select the structure using the question, not this error")
    return f"Answer schema error: {detail}. Use the requested public structure directly, " \
           "without a task-name wrapper. This is format feedback, not confirmation of clinical correctness."
