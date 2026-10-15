"""Deterministic, record-grounded answers for clinical chart-review questions."""

from collections import defaultdict
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_EVEN
from functools import cached_property
import weakref

from dataset.common import canonical, codes, observation_time, patient_ref, resource_id, timestamp

LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
UCUM = "http://unitsofmeasure.org"
OBS_STATUSES = {"final", "amended", "corrected"}
PATIENT_TYPES = {"Encounter", "Condition", "Procedure", "MedicationRequest", "Observation", "DiagnosticReport"}
GLYCEMIC_CODES = {(SNOMED, "44054006"), (SNOMED, "714628002")}


class InvalidExample(ValueError):
    pass


class IneligibleExample(InvalidExample):
    """The chart has no clinical anchor for this question family."""


def number(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise InvalidExample("Non-finite numeric value")
    return result


def formatted(value):
    text = format(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_EVEN), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def obs_matches(resource, code):
    return (LOINC, code) in codes(resource.get("code", {})) or any(
        (LOINC, code) in codes(c.get("code", {})) for c in resource.get("component", []))


def quantity(resource, code, unit):
    if (LOINC, code) in codes(resource.get("code", {})):
        value = resource.get("valueQuantity")
    else:
        components = [c for c in resource.get("component", []) if (LOINC, code) in codes(c.get("code", {}))]
        if len(components) != 1:
            raise InvalidExample("Missing or ambiguous measurement component")
        value = components[0].get("valueQuantity")
    if value is None:
        return None  # A matching observation exists but has no numeric quantity.
    if value.get("comparator") or value.get("system") != UCUM or "value" not in value:
        raise InvalidExample("Censored or malformed quantity")
    amount = number(value["value"])
    if value.get("code") == unit:
        return amount
    if unit == "kg" and value.get("code") == "[lb_av]":
        return amount * Decimal("0.45359237")
    if unit == "/min" and value.get("code") == "{beats}/min":
        return amount
    raise InvalidExample(f"Unsupported unit conversion: {value.get('code')} -> {unit}")


def concept(value):
    coding = sorted((c for c in value.get("coding", []) if c.get("system") and c.get("code")),
                    key=lambda c: (c["system"], c["code"]))
    if not coding:
        raise InvalidExample("Clinical concept has no coded identifier")
    c = coding[0]
    return {"system": c["system"], "code": c["code"],
            "name": c.get("display") or value.get("text") or c["code"]}


def link(resource):
    return resource.get("encounter", {}).get("reference")


def pick(resources, when, latest=True):
    """Never break a clinically ambiguous timestamp tie by arbitrary UUID order."""
    if not resources:
        raise IneligibleExample("No qualifying clinical anchor")
    dated = [(when(r), r) for r in resources]
    target = (max if latest else min)(t for t, _ in dated)
    selected = [r for t, r in dated if t == target]
    if len(selected) != 1:
        raise InvalidExample("Ambiguous anchor timestamp")
    return selected[0]


def start(encounter):
    return timestamp(encounter.get("period", {}).get("start"))


def end(encounter):
    return timestamp(encounter.get("period", {}).get("end"))


def procedure_time(procedure):
    return timestamp(procedure.get("performedDateTime") or procedure.get("performedPeriod", {}).get("start"))


def report_time(report):
    return timestamp(report.get("effectiveDateTime") or report.get("effectivePeriod", {}).get("start"))


class Chart:
    def __init__(self, snapshot, patient):
        if patient not in snapshot.patients:
            raise InvalidExample(f"Unknown patient: {patient}")
        self.snapshot, self.patient = weakref.proxy(snapshot), patient
        self._resources, self._linked, self._resolved = {}, {}, {}

    def resources(self, kind):
        if kind not in self._resources:
            self._resources[kind] = self.snapshot.for_patient(self.patient, kind)
            self._resolved.update({resource_id(r): r for r in self._resources[kind]})
        return self._resources[kind]

    def linked(self, kind, encounter):
        if kind not in self._linked:
            grouped = defaultdict(list)
            for r in self.resources(kind):
                grouped[link(r)].append(r)
            self._linked[kind] = grouped
        return self._linked[kind].get(resource_id(encounter), [])

    def resolve(self, reference, kind, parent=None):
        if reference.startswith("#") and parent:
            matches = [r for r in parent.get("contained", []) if r.get("id") == reference[1:]]
            r = matches[0] if len(matches) == 1 else None
        else:
            r = self._resolved.get(reference) or self.snapshot.resources.get(reference)
        if not r or r.get("resourceType") != kind:
            raise InvalidExample(f"Unresolved or incorrectly typed reference: {reference}")
        owner = patient_ref(r)
        if owner and owner != self.patient:
            raise InvalidExample(f"Reference crosses patient boundary: {reference}")
        if not reference.startswith("#"):
            self._resolved[reference] = r
        return r

    def encounter(self, spec, mature=False, rank=0):
        cutoff = timestamp(spec["as_of"])
        available = [r for r in self.resources("Encounter") if r.get("status") == "finished"
                     and r.get("class", {}).get("code") == "IMP" and r.get("period", {}).get("end")
                     and start(r) <= end(r) < cutoff
                     and (not mature or end(r) + timedelta(days=spec.get("followup_days", 30)) <= cutoff)]
        if spec.get("inpatient_orders_required", spec.get("operation") in {"inpatient_medications", "followup_medication_comparison"}):
            available = [r for r in available if self.orders(r, set(), cutoff)]
        selected = None
        for _ in range(rank + 1):
            selected = pick(available, start, latest=spec.get("operation") != "subsequent_admission")
            available = [r for r in available if resource_id(r) != resource_id(selected)]
        return selected

    def medication(self, order, evidence):
        evidence.add(resource_id(order))
        if order.get("medicationCodeableConcept"):
            return concept(order["medicationCodeableConcept"])
        ref = order.get("medicationReference", {}).get("reference", "")
        medication = self.resolve(ref, "Medication", order)
        if not ref.startswith("#"):
            evidence.add(resource_id(medication))
        return concept(medication.get("code", {}))

    def medication_requests(self, cutoff, encounter=None):
        source = self.resources("MedicationRequest") if encounter is None else self.linked("MedicationRequest", encounter)
        result = []
        for r in source:
            if r.get("status") in {"entered-in-error", "draft"} or r.get("intent") != "order":
                continue
            if timestamp(r.get("authoredOn")) >= cutoff:
                continue
            result.append(r)
        return result

    def condition_reasons(self, resource):
        """Resolve explicit Condition reasons; other reason resource types are excluded."""
        result = []
        for reason in resource.get("reasonReference", []):
            ref = reason.get("reference", "")
            if ref.startswith("#"):
                contained = [r for r in resource.get("contained", []) if r.get("id") == ref[1:]]
                if len(contained) == 1 and contained[0].get("resourceType") != "Condition":
                    continue
            elif not ref.startswith("Condition/"):
                continue
            condition = self.resolve(ref, "Condition", resource)
            if any(code == "entered-in-error" for _, code in codes(condition.get("verificationStatus", {}))):
                continue
            result.append((ref, condition))
        return result

    def indications(self, resource, evidence):
        result = []
        for ref, condition in self.condition_reasons(resource):
            if not ref.startswith("#"):
                evidence.add(resource_id(condition))
            result.append(concept(condition.get("code", {})))
        return sorted(result, key=canonical)

    def orders(self, encounter, evidence, cutoff):
        result = []
        for r in self.medication_requests(cutoff, encounter):
            result.append({"medication": self.medication(r, evidence),
                           "ordered_at": timestamp(r.get("authoredOn")).isoformat(),
                           "recorded_status": r.get("status", "unknown")})
        return sorted(result, key=canonical)

    def conditions(self, encounter, evidence):
        result = []
        for r in self.linked("Condition", encounter):
            if any(code == "entered-in-error" for _, code in codes(r.get("verificationStatus", {}))):
                continue
            evidence.add(resource_id(r))
            result.append({"condition": concept(r.get("code", {})), "onset": r.get("onsetDateTime")})
        return sorted(result, key=canonical)

    def observations(self, code, lower, upper, encounter=None):
        source = self.resources("Observation") if encounter is None else self.linked("Observation", encounter)
        return [r for r in source if r.get("status") in OBS_STATUSES and obs_matches(r, code)
                and lower <= observation_time(r) < upper]

    def measurement(self, resources, code, unit, evidence, latest=True):
        if not resources:
            return {"status": "no_matching_record"}
        times = [observation_time(r) for r in resources]
        when = (max if latest else min)(times)
        selected = [r for r in resources if observation_time(r) == when]
        values = {quantity(r, code, unit) for r in selected}
        if len(values) != 1:
            raise InvalidExample("Conflicting values at the selected measurement time")
        evidence.update(resource_id(r) for r in selected)
        value = next(iter(values))
        return {"status": "missing_value" if value is None else "found", "time": when.isoformat(),
                "value": None if value is None else format(value, "f"), "unit": unit}

    def cbc_reports(self, spec):
        cutoff = timestamp(spec["as_of"])
        return [r for r in self.resources("DiagnosticReport") if r.get("status") in OBS_STATUSES
                and (LOINC, "58410-2") in codes(r.get("code", {})) and report_time(r) < cutoff]

    def cbc(self, spec, evidence, selected=None):
        if selected is None:
            selected = pick(self.cbc_reports(spec), report_time)
        evidence.add(resource_id(selected))
        observations = []
        for ref in selected.get("result", []):
            r = self.resolve(ref.get("reference", ""), "Observation", selected)
            # Contained report observations need their report as evidence, not a fake global ID.
            if ref.get("reference", "").startswith("#"):
                raise InvalidExample("Contained report results are not supported in this snapshot")
            if r.get("status") in OBS_STATUSES and observation_time(r) < timestamp(spec["as_of"]):
                observations.append(r)
        result = {"report_time": report_time(selected).isoformat()}
        for name, code, unit in [("hemoglobin", "718-7", "g/dL"), ("platelets", "777-3", "10*3/uL")]:
            found = [r for r in observations if obs_matches(r, code)]
            result[name] = self.measurement(found, code, unit, evidence)
        return result

    def followup(self, admission, evidence, days=30):
        discharge = end(admission)
        eligible = [r for r in self.resources("Encounter") if r.get("status") == "finished"
                    and r.get("class", {}).get("code") == "AMB"
                    and discharge < start(r) <= discharge + timedelta(days=days)]
        if not eligible:
            return None
        selected = pick(eligible, start, latest=False)
        evidence.add(resource_id(selected))
        return selected

    @cached_property
    def glycemic_condition(self):
        return pick([r for r in self.resources("Condition") if codes(r.get("code", {})) & GLYCEMIC_CODES
                     and r.get("onsetDateTime")], lambda r: timestamp(r["onsetDateTime"]), latest=False)


def chart_for(snapshot, patient):
    if getattr(snapshot, "_clinical_patient", None) != patient:
        snapshot._clinical_patient = patient
        snapshot._clinical_chart = Chart(snapshot, patient)
    return snapshot._clinical_chart


def compare_products(first, second):
    def keyed(rows):
        return {(r["medication"]["system"], r["medication"]["code"]): r["medication"] for r in rows}
    a, b = keyed(first), keyed(second)
    return {"both": [a[k] for k in sorted(a.keys() & b.keys())],
            "only_first": [a[k] for k in sorted(a.keys() - b.keys())],
            "only_second": [b[k] for k in sorted(b.keys() - a.keys())]}


def difference(before, after):
    if before.get("status") != "found" or after.get("status") != "found":
        return None
    return formatted(number(after["value"]) - number(before["value"]))


def evaluate(snapshot, patient, spec):
    chart = chart_for(snapshot, patient)
    operation, cutoff = spec["operation"], timestamp(spec["as_of"])
    window = spec.get("followup_days", 365 if operation in {"renal_followup", "glycemic_followup"} else 30)
    evidence = set()
    if operation in {"inpatient_conditions", "inpatient_medications", "preadmission_treated_conditions",
                     "discharge_followup", "admission_medication_comparison",
                     "followup_medication_comparison", "subsequent_admission"}:
        mature = operation in {"discharge_followup", "followup_medication_comparison"}
        admission = chart.encounter(spec, mature=mature)
        evidence.add(resource_id(admission))
        answer = {"admitted_at": start(admission).isoformat(), "discharged_at": end(admission).isoformat()}
        if operation == "inpatient_conditions":
            answer["conditions"] = chart.conditions(admission, evidence)
        elif operation == "inpatient_medications":
            answer["orders"] = chart.orders(admission, evidence, cutoff)
        elif operation == "preadmission_treated_conditions":
            admitted = start(admission)
            lower = admitted - timedelta(days=spec.get("lookback_days", 365))
            selected = {}
            for order in chart.medication_requests(admitted):
                if timestamp(order["authoredOn"]) < lower:
                    continue
                for ref, condition in chart.condition_reasons(order):
                    if not condition.get("onsetDateTime"):
                        continue
                    onset = timestamp(condition["onsetDateTime"])
                    abated = condition.get("abatementDateTime")
                    if onset >= admitted or (abated and timestamp(abated) < admitted):
                        continue
                    key = (resource_id(order), ref) if ref.startswith("#") else ref
                    if not ref.startswith("#"):
                        evidence.add(resource_id(condition))
                    entry = selected.setdefault(key, {"condition": concept(condition.get("code", {})),
                        "onset": onset.isoformat(), "abatement": timestamp(abated).isoformat() if abated else None,
                        "orders": []})
                    row = {"medication": chart.medication(order, evidence),
                           "ordered_at": timestamp(order["authoredOn"]).isoformat()}
                    if row not in entry["orders"]:
                        entry["orders"].append(row)
            for entry in selected.values():
                entry["orders"].sort(key=canonical)
            answer["conditions_with_prior_orders"] = sorted(selected.values(), key=canonical)
        elif operation == "admission_medication_comparison":
            previous = chart.encounter(spec, rank=1)
            evidence.add(resource_id(previous))
            answer["previous_admitted_at"] = start(previous).isoformat()
            answer["products"] = compare_products(chart.orders(previous, evidence, cutoff), chart.orders(admission, evidence, cutoff))
        elif operation in {"discharge_followup", "followup_medication_comparison"}:
            # Orders establish this family's eligible anchor even when no follow-up exists.
            inpatient_orders = chart.orders(admission, evidence, cutoff) if operation == "followup_medication_comparison" else None
            followup = chart.followup(admission, evidence, window)
            answer["followup"] = None
            if followup:
                answer["followup"] = {"started_at": start(followup).isoformat(),
                                      "days_after_discharge": formatted(number((start(followup) - end(admission)).total_seconds()) / 86400)}
                if operation == "discharge_followup":
                    answer["followup"]["conditions"] = chart.conditions(followup, evidence)
                else:
                    answer["products"] = compare_products(inpatient_orders, chart.orders(followup, evidence, cutoff))
        else:
            subsequent = [r for r in chart.resources("Encounter") if r.get("status") == "finished"
                          and r.get("class", {}).get("code") == "IMP"
                          and end(admission) < start(r) < cutoff]
            answer["subsequent_admission"] = None
            if subsequent:
                selected = pick(subsequent, start, latest=False)
                evidence.add(resource_id(selected))
                first = chart.conditions(admission, evidence)
                second = chart.conditions(selected, evidence)
                a = {(x["condition"]["system"], x["condition"]["code"]): x["condition"] for x in first}
                b = {(x["condition"]["system"], x["condition"]["code"]) for x in second}
                answer["subsequent_admission"] = {"admitted_at": start(selected).isoformat(),
                                         "days_after_discharge": formatted(number((start(selected) - end(admission)).total_seconds()) / 86400),
                                         "conditions": second,
                                         "shared_conditions": [a[k] for k in sorted(a.keys() & b)]}
    elif operation in {"cbc_panel", "cbc_change"}:
        if operation == "cbc_change":
            reports = chart.cbc_reports(spec)
            interval = timedelta(days=spec.get("comparison_days", 90))
            def earlier(report):
                return [r for r in reports if report_time(report) - interval <= report_time(r) < report_time(report)]
            latest_report = pick([r for r in reports if earlier(r)], report_time)
            previous_report = pick(earlier(latest_report), report_time)
            latest = chart.cbc(spec, evidence, latest_report)
            previous = chart.cbc(spec, evidence, previous_report)
            answer = {"previous": previous, "latest": latest,
                      "hemoglobin_change_g_dL": difference(previous["hemoglobin"], latest["hemoglobin"]),
                      "platelet_change_10_3_uL": difference(previous["platelets"], latest["platelets"])}
        else:
            answer = chart.cbc(spec, evidence)
    elif operation == "outpatient_bp":
        encounters = [r for r in chart.resources("Encounter") if r.get("status") == "finished"
                      and r.get("class", {}).get("code") == "AMB" and start(r) < cutoff
                      and chart.observations("8480-6", start(r), cutoff, r)]
        selected = pick(encounters, start)
        evidence.add(resource_id(selected))
        answer = {"visit_started_at": start(selected).isoformat()}
        for name, code in [("systolic", "8480-6"), ("diastolic", "8462-4")]:
            found = chart.observations(code, start(selected), cutoff, selected)
            answer[name] = chart.measurement(found, code, "mm[Hg]", evidence)
    elif operation == "procedure_indications":
        eligible = []
        for encounter in chart.resources("Encounter"):
            if encounter.get("status") != "finished" or start(encounter) >= cutoff:
                continue
            procedures = [r for r in chart.linked("Procedure", encounter)
                          if r.get("status") == "completed" and procedure_time(r) < cutoff
                          and chart.condition_reasons(r)]
            if procedures:
                eligible.append(encounter)
        encounter = pick(eligible, start)
        procedures = [r for r in chart.linked("Procedure", encounter)
                      if r.get("status") == "completed" and procedure_time(r) < cutoff
                      and chart.condition_reasons(r)]
        evidence.update(resource_id(r) for r in procedures)
        evidence.add(resource_id(encounter))
        answer = {"encounter_started_at": start(encounter).isoformat(),
                  "procedures": sorted([{"procedure": concept(r.get("code", {})),
                                         "performed_at": procedure_time(r).isoformat(),
                                         "indications": chart.indications(r, evidence)}
                                        for r in procedures], key=canonical)}
    elif operation == "medication_indications":
        available = []
        for encounter in chart.resources("Encounter"):
            if encounter.get("status") != "finished" or start(encounter) >= cutoff:
                continue
            if any(chart.condition_reasons(r) for r in chart.medication_requests(cutoff, encounter)):
                available.append(encounter)
        selected = pick(available, start)
        evidence.add(resource_id(selected))
        answer = {"encounter_started_at": start(selected).isoformat(), "orders": []}
        for order in chart.medication_requests(cutoff, selected):
            if not chart.condition_reasons(order):
                continue
            answer["orders"].append({"medication": chart.medication(order, evidence),
                                     "indications": chart.indications(order, evidence)})
        answer["orders"].sort(key=canonical)
    elif operation == "renal_followup":
        candidates = []
        for r in chart.resources("Observation"):
            if r.get("status") not in OBS_STATUSES or not obs_matches(r, "2160-0") or not link(r):
                continue
            encounter = chart.resolve(link(r), "Encounter")
            if encounter.get("status") == "finished" and encounter.get("class", {}).get("code") == "AMB":
                candidates.append(r)
        baseline = chart.measurement(candidates, "2160-0", "mg/dL", evidence, latest=False)
        if baseline["status"] == "no_matching_record":
            raise IneligibleExample("No ambulatory creatinine baseline")
        anchor = timestamp(baseline["time"])
        if anchor + timedelta(days=window) > cutoff:
            raise IneligibleExample("Ambulatory creatinine baseline has not completed its follow-up window")
        for r in candidates:
            if observation_time(r) == anchor:
                evidence.add(link(r))
        following = chart.observations("2160-0", anchor + timedelta(microseconds=1), anchor + timedelta(days=window))
        followup = chart.measurement(following, "2160-0", "mg/dL", evidence, latest=False)
        answer = {"baseline": baseline, "followup": followup, "change_mg_dL": difference(baseline, followup)}
    elif operation == "glycemic_followup":
        condition = chart.glycemic_condition
        anchor = timestamp(condition["onsetDateTime"])
        if anchor + timedelta(days=window) > cutoff:
            raise IneligibleExample("Glycemic diagnosis has not completed its follow-up window")
        evidence.add(resource_id(condition))
        answer = {"condition": concept(condition.get("code", {})), "onset": anchor.isoformat()}
        following = chart.observations("4548-4", anchor + timedelta(microseconds=1), anchor + timedelta(days=window))
        answer["followup"] = chart.measurement(following, "4548-4", "%", evidence, latest=False)
        if following:
            baseline = chart.observations("4548-4", anchor - timedelta(days=90), anchor + timedelta(microseconds=1))
            answer["baseline"] = chart.measurement(baseline, "4548-4", "%", evidence)
            answer["change_percentage_points"] = difference(answer["baseline"], answer["followup"])
    else:
        raise InvalidExample(f"Unsupported clinical operation: {operation}")
    return answer, sorted(evidence)
