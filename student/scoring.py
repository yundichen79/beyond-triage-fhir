"""Compare structured answers without an LLM judge or feedback to the agent."""

from decimal import Decimal, DecimalException
import re

from dataset.common import timestamp


UNIT_ALIASES = {"mmHg": "mm[Hg]", "10^3/uL": "10*3/uL", "10³/µL": "10*3/uL"}
NUMERIC = {"value", "days_after_discharge", "change_percentage_points", "change_mg_dL",
           "hemoglobin_change_g_dL", "platelet_change_10_3_uL"}
TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$")


def equal(actual, expected, key=""):
    if expected is None or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        # Coded identity determines the clinical concept; display wording can vary.
        if {"system", "code"} <= expected.keys():
            return actual.get("system") == expected["system"] and actual.get("code") == expected["code"]
        return actual.keys() == expected.keys() and all(equal(actual[k], v, k) for k, v in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        # A matching preserves multiplicity while ignoring ordering, including
        # the rare case of values within tolerance of more than one candidate.
        matches = {}
        def assign(i, visited):
            for j, value in enumerate(actual):
                if j in visited or not equal(value, expected[i], key):
                    continue
                visited.add(j)
                if j not in matches or assign(matches[j], visited):
                    matches[j] = i
                    return True
            return False
        return all(assign(i, set()) for i in range(len(expected)))
    if key in NUMERIC:
        if isinstance(actual, bool):
            return False
        try:
            a, b = Decimal(str(actual)), Decimal(str(expected))
            return a.is_finite() and b.is_finite() and abs(a - b) <= Decimal("0.000001")
        except DecimalException:
            return False
    if isinstance(expected, str) and TIME.fullmatch(expected):
        try:
            return isinstance(actual, str) and bool(TIME.fullmatch(actual)) and timestamp(actual) == timestamp(expected)
        except ValueError:
            return False
    if key == "unit":
        return isinstance(actual, str) and UNIT_ALIASES.get(actual, actual) == UNIT_ALIASES.get(expected, expected)
    return type(actual) is type(expected) and actual == expected


def score_episode(row, episode):
    final = episode["final"]
    required = set(row["ground_truth_resource_ids"])
    retrieved = set(episode["retrieved_resource_ids"])
    cited = set(final["evidence"]) if final else set()
    correct = final is not None and equal(final["answer"], row["ground_truth_answer"])
    missing = sorted(required - retrieved)
    return {"answer_correct": correct, "required_evidence_retrieved": not missing,
            "answer_correct_with_evidence": correct and not missing and bool(cited) and required <= cited <= retrieved,
            "evidence_recall": len(required & retrieved) / len(required) if required else 1.0,
            "citation_evidence_recall": len(required & cited) / len(required) if required else 1.0,
            "missing_evidence": missing}
