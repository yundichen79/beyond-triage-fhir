"""Offline audit of the preserved cohort; never modifies dataset or answer rules.

Independent selection/answer implementation below does not call the production
oracle. Raw reference normalization, schema checking and the separately named
canonical replay reuse repository infrastructure; this is not human review.
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

from dataset.review import source_snapshot
from dataset.validate import validate_rows
from pilot.contract import answer_error, contract_for_question

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dataset-audit"
LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
FINAL = {"final", "amended", "corrected"}


def dt(x):
    return datetime.fromisoformat(x.replace("Z", "+00:00")).replace(tzinfo=timezone.utc) if len(x) == 10 else datetime.fromisoformat(x.replace("Z", "+00:00")).astimezone(timezone.utc)


def rid(r):
    return r["resourceType"] + "/" + r["id"]


def codes(r):
    return {(c.get("system"), c.get("code")) for c in r.get("coding", [])}


def concept(c):
    v = sorted((v for v in c["coding"] if v.get("code") and v.get("system")), key=lambda v: (v["system"], v["code"]))[0]
    return dict(system=v["system"], code=v["code"], name=v.get("display") or c.get("text") or v["code"])


def time(r):
    return dt(r.get("effectiveDateTime") or r.get("effectivePeriod", {}).get("start") or r.get("performedDateTime") or r.get("performedPeriod", {}).get("start") or r.get("period", {}).get("start"))


def end(r):
    return dt(r["period"]["end"])


def linked(r, e):
    return r.get("encounter", {}).get("reference") == rid(e)


def round1(x):
    return format(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_EVEN), "f")


def normalize(x):
    # Independent equality: preserve all fields, normalize numeric strings and times,
    # and ignore ordering while preserving array multiplicity.
    if isinstance(x, dict):
        return {k: normalize(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return sorted((normalize(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(x, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", x):
        return format(Decimal(x).normalize(), "f")
    if isinstance(x, str) and re.match(r"\d{4}-\d\d-\d\dT", x):
        return dt(x).isoformat()
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return format(Decimal(str(x)).normalize(), "f")
    return x


class IndependentChart:
    def __init__(self, snapshot, patient):
        self.all = snapshot.resources
        self.patient = patient
        self.groups = {k: list(snapshot.for_patient(patient, k)) for k in
                       ("Encounter", "Condition", "Observation", "DiagnosticReport", "MedicationRequest", "Procedure")}
        self.evidence = set()

    def add(self, r):
        self.evidence.add(rid(r))
        return r

    def resolve(self, ref, parent):
        if ref.startswith("#"):
            return next(r for r in parent["contained"] if r["id"] == ref[1:])
        r = self.all[ref]
        owner = (r.get("subject") or r.get("patient") or {}).get("reference")
        assert owner in (None, self.patient), "Cross-patient resolved reference"
        return self.add(r)

    def choose(self, rows, key=time, latest=True):
        rows = sorted(rows, key=key, reverse=latest)
        assert rows, "Missing independently eligible anchor"
        assert len(rows) == 1 or key(rows[0]) != key(rows[1]), "Ambiguous independently selected anchor"
        return self.add(rows[0])

    def reasons(self, r, capture=False):
        found = []
        for entry in r.get("reasonReference", []):
            ref = entry["reference"]
            c = (next((x for x in r.get("contained", []) if x["id"] == ref[1:]), None)
                 if ref.startswith("#") else self.all.get(ref))
            if c and c["resourceType"] == "Condition" and not any(code == "entered-in-error" for _, code in codes(c.get("verificationStatus", {}))):
                if capture and not ref.startswith("#"):
                    self.add(c)
                found.append(c)
        return found

    def orders(self, cutoff, encounter=None):
        return [r for r in self.groups["MedicationRequest"] if r.get("intent") == "order"
                and r.get("status") not in {"draft", "entered-in-error"}
                and dt(r["authoredOn"]) < cutoff and (encounter is None or linked(r, encounter))]

    def drug(self, r):
        self.add(r)
        if r.get("medicationCodeableConcept"):
            return concept(r["medicationCodeableConcept"])
        return concept(self.resolve(r["medicationReference"]["reference"], r)["code"])

    def conditions(self, e):
        rows = [r for r in self.groups["Condition"] if linked(r, e)
                and not any(c == "entered-in-error" for _, c in codes(r.get("verificationStatus", {})))]
        return [dict(condition=concept(self.add(r)["code"]), onset=r.get("onsetDateTime")) for r in rows]

    def observations(self, code):
        return [r for r in self.groups["Observation"] if r.get("status") in FINAL
                and ((LOINC, code) in codes(r.get("code", {})) or any((LOINC, code) in codes(c.get("code", {})) for c in r.get("component", [])))]

    def measurement(self, rows, code, unit, latest=True):
        if not rows:
            return {"status": "no_matching_record"}
        selected_time = (max if latest else min)(time(r) for r in rows)
        chosen = [r for r in rows if time(r) == selected_time]
        values = set()
        for r in chosen:
            self.add(r)
            node = r if (LOINC, code) in codes(r.get("code", {})) else next(c for c in r["component"] if (LOINC, code) in codes(c.get("code", {})))
            q = node.get("valueQuantity")
            if q is None:
                values.add(None)
            else:
                assert q.get("system") == "http://unitsofmeasure.org" and q.get("code") == unit and not q.get("comparator"), "Unsupported audited measurement units"
                values.add(Decimal(str(q["value"])))
        assert len(values) == 1, "Conflicting measurements"
        value = next(iter(values))
        return dict(status="missing_value" if value is None else "found", time=selected_time.isoformat(), value=None if value is None else str(value), unit=unit)

    def delta(self, before, after):
        return round1(Decimal(after["value"]) - Decimal(before["value"])) if before.get("status") == after.get("status") == "found" else None

    def cbc(self, report, cutoff):
        self.add(report)
        results = [self.all[x["reference"]] for x in report.get("result", [])]
        answer = {"report_time": time(report).isoformat()}
        for field, code, unit in (("hemoglobin", "718-7", "g/dL"), ("platelets", "777-3", "10*3/uL")):
            eligible_ids = {rid(r) for r in self.observations(code)}
            answer[field] = self.measurement([r for r in results if rid(r) in eligible_ids and time(r) < cutoff], code, unit)
        return answer

    def products(self, a, b):
        first = {(c["system"], c["code"]): c for c in (self.drug(r) for r in a)}
        second = {(c["system"], c["code"]): c for c in (self.drug(r) for r in b)}
        return dict(both=[first[k] for k in first.keys() & second.keys()],
                    only_first=[first[k] for k in first.keys() - second.keys()],
                    only_second=[second[k] for k in second.keys() - first.keys()])

    def answer(self, row):
        f = row["family"]
        spec = row["oracle_spec"]
        cutoff = dt(spec["as_of"])
        window = timedelta(days=spec.get("followup_days", 365))
        encounters = [r for r in self.groups["Encounter"] if r.get("status") == "finished" and time(r) < cutoff]
        stays = [r for r in encounters if r.get("class", {}).get("code") == "IMP" and r.get("period", {}).get("end") and time(r) <= end(r) < cutoff]
        stay_families = {"inpatient_conditions", "inpatient_medications", "preadmission_treated_conditions", "admission_medication_comparison", "discharge_followup", "followup_medication_comparison", "subsequent_admission"}
        if f in stay_families:
            if f in {"discharge_followup", "followup_medication_comparison"}:
                stays = [r for r in stays if end(r) + window <= cutoff]
            if f in {"inpatient_medications", "followup_medication_comparison"}:
                stays = [r for r in stays if self.orders(cutoff, r)]
            stay = self.choose(stays, latest=f != "subsequent_admission")
            a = dict(admitted_at=time(stay).isoformat(), discharged_at=end(stay).isoformat())
            if f == "inpatient_conditions":
                a["conditions"] = self.conditions(stay)
            elif f == "inpatient_medications":
                a["orders"] = [dict(medication=self.drug(r), ordered_at=dt(r["authoredOn"]).isoformat(), recorded_status=r.get("status", "unknown")) for r in self.orders(cutoff, stay)]
            elif f == "admission_medication_comparison":
                prev = self.choose([r for r in stays if rid(r) != rid(stay)])
                a.update(previous_admitted_at=time(prev).isoformat(), products=self.products(self.orders(cutoff, prev), self.orders(cutoff, stay)))
            elif f == "preadmission_treated_conditions":
                found = {}
                for order in self.orders(time(stay)):
                    if dt(order["authoredOn"]) < time(stay) - timedelta(days=365):
                        continue
                    for c in self.reasons(order):
                        onset, abatement = c.get("onsetDateTime"), c.get("abatementDateTime")
                        if not onset or dt(onset) >= time(stay) or abatement and dt(abatement) < time(stay):
                            continue
                        self.add(c)
                        entry = found.setdefault(rid(c), dict(condition=concept(c["code"]), onset=dt(onset).isoformat(), abatement=dt(abatement).isoformat() if abatement else None, orders=[]))
                        item = dict(medication=self.drug(order), ordered_at=dt(order["authoredOn"]).isoformat())
                        if item not in entry["orders"]:
                            entry["orders"].append(item)
                a["conditions_with_prior_orders"] = list(found.values())
            elif f in {"discharge_followup", "followup_medication_comparison"}:
                first_orders = self.orders(cutoff, stay)
                if f == "followup_medication_comparison":
                    for r in first_orders:
                        self.drug(r)
                candidates = [r for r in encounters if r.get("class", {}).get("code") == "AMB" and end(stay) < time(r) <= end(stay) + window]
                a["followup"] = None
                if candidates:
                    follow = self.choose(candidates, latest=False)
                    a["followup"] = dict(started_at=time(follow).isoformat(), days_after_discharge=round1(Decimal(str((time(follow)-end(stay)).total_seconds())) / 86400))
                    if f == "discharge_followup":
                        a["followup"]["conditions"] = self.conditions(follow)
                    else:
                        a["products"] = self.products(first_orders, self.orders(cutoff, follow))
            else:
                candidates = [r for r in encounters if r.get("class", {}).get("code") == "IMP" and time(r) > end(stay)]
                a["subsequent_admission"] = None
                if candidates:
                    follow = self.choose(candidates, latest=False)
                    first = { (x["condition"]["system"], x["condition"]["code"]): x["condition"] for x in self.conditions(stay)}
                    second = self.conditions(follow)
                    second_codes = {(x["condition"]["system"], x["condition"]["code"]) for x in second}
                    a["subsequent_admission"] = dict(admitted_at=time(follow).isoformat(), conditions=second, days_after_discharge=round1(Decimal(str((time(follow)-end(stay)).total_seconds()))/86400), shared_conditions=[first[k] for k in first.keys() & second_codes])
            return a
        if f in {"cbc_panel", "cbc_change"}:
            reports = [r for r in self.groups["DiagnosticReport"] if r.get("status") in FINAL and (LOINC,"58410-2") in codes(r.get("code",{})) and time(r) < cutoff]
            if f == "cbc_panel":
                return self.cbc(self.choose(reports), cutoff)
            pairs = [(b,a) for b in reports for a in reports if timedelta(0) < time(b)-time(a) <= timedelta(days=90)]
            latest = self.choose(list({rid(b):b for b,a in pairs}.values()))
            previous = self.choose([a for b,a in pairs if rid(b)==rid(latest)])
            first,second = self.cbc(previous,cutoff), self.cbc(latest,cutoff)
            return dict(previous=first, latest=second, hemoglobin_change_g_dL=self.delta(first["hemoglobin"],second["hemoglobin"]), platelet_change_10_3_uL=self.delta(first["platelets"],second["platelets"]))
        if f == "outpatient_bp":
            systolic = self.observations("8480-6")
            eligible = [e for e in encounters if e.get("class",{}).get("code")=="AMB" and any(linked(r,e) and time(e)<=time(r)<cutoff for r in systolic)]
            e=self.choose(eligible)
            return dict(visit_started_at=time(e).isoformat(), **{field:self.measurement([r for r in self.observations(code) if linked(r,e) and time(e)<=time(r)<cutoff],code,"mm[Hg]") for field,code in (("systolic","8480-6"),("diastolic","8462-4"))})
        if f in {"procedure_indications", "medication_indications"}:
            source = ([r for r in self.groups["Procedure"] if r.get("status")=="completed" and time(r)<cutoff] if f=="procedure_indications" else self.orders(cutoff))
            source = [r for r in source if self.reasons(r)]
            e=self.choose([e for e in encounters if any(linked(r,e) for r in source)])
            items=[]
            for r in source:
                if not linked(r,e):continue
                self.add(r)
                item=dict(indications=[concept(c["code"]) for c in self.reasons(r,True)])
                if f=="procedure_indications":item.update(procedure=concept(r["code"]),performed_at=time(r).isoformat())
                else:item["medication"]=self.drug(r)
                items.append(item)
            return dict(encounter_started_at=time(e).isoformat(), **{("procedures" if f=="procedure_indications" else "orders"):items})
        if f=="glycemic_followup":
            c=self.choose([r for r in self.groups["Condition"] if r.get("onsetDateTime") and codes(r.get("code",{})) & {(SNOMED,"44054006"),(SNOMED,"714628002")}],key=lambda r:dt(r["onsetDateTime"]),latest=False)
            anchor=dt(c["onsetDateTime"])
            assert anchor+window<=cutoff
            observations=self.observations("4548-4")
            follow=self.measurement([r for r in observations if anchor<time(r)<anchor+window],"4548-4","%",False)
            a=dict(condition=concept(c["code"]),onset=anchor.isoformat(),followup=follow)
            if follow["status"]!="no_matching_record":
                a["baseline"]=self.measurement([r for r in observations if anchor-timedelta(days=90)<=time(r)<=anchor],"4548-4","%")
                a["change_percentage_points"]=self.delta(a["baseline"],follow)
            return a
        if f=="renal_followup":
            observations=self.observations("2160-0")
            ambulatory={rid(e) for e in encounters if e.get("class",{}).get("code")=="AMB"}
            eligible=[r for r in observations if r.get("encounter",{}).get("reference") in ambulatory]
            baseline=self.measurement(eligible,"2160-0","mg/dL",False)
            anchor=dt(baseline["time"])
            assert anchor+window<=cutoff
            for r in eligible:
                if time(r)==anchor:self.evidence.add(r["encounter"]["reference"])
            follow=self.measurement([r for r in observations if anchor<time(r)<anchor+window],"2160-0","mg/dL",False)
            return dict(baseline=baseline,followup=follow,change_mg_dL=self.delta(baseline,follow))
        raise ValueError(f)


def main():
    paths=[ROOT/"data/processed/cohort-200"/n for n in ("questions.jsonl","patient_splits.json")]
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows=[json.loads(l) for l in paths[0].read_text().splitlines()]
    splits=json.loads(paths[1].read_text()); config=json.loads((ROOT/"configs/cohort.json").read_text())
    grouped=defaultdict(list)
    for r in rows:grouped[r["patient_id"]].append(r)
    result=dict(dataset_sha256=before,questions=len(rows),patients=len(grouped),checks=[],errors=[],warnings=defaultdict(list),coverage={},canonical_replay_errors=[])
    for field, keys in (("unique_ids",[r["id"] for r in rows]),("unique_patient_family",[(r["patient_id"],r["family"]) for r in rows])):
        if len(keys)!=len(set(keys)):result["errors"].append(field)
    for i,(patient,group) in enumerate(sorted(grouped.items()),1):
        snapshot=source_snapshot(ROOT/"data/raw/batch-001/fhir"/(patient.split('/')[1]+'.json'))
        valid=validate_rows(snapshot,group,{patient:splits[patient]},config)
        result["canonical_replay_errors"].extend(valid["errors"])
        for row in group:
            check=dict(id=row['id'],family=row['family'],split=row['split'])
            try:
                name,shape=contract_for_question(row['question'])
                assert answer_error(row['ground_truth_answer'],shape) is None,'Public schema rejects expected answer'
                assert row['split']==splits[patient]
                chart=IndependentChart(snapshot,patient)
                answer=chart.answer(row)
                check['answer_matches']=normalize(answer)==normalize(row['ground_truth_answer'])
                check['evidence_matches']=chart.evidence==set(row['ground_truth_resource_ids'])
                if not check['answer_matches']:
                    result['errors'].append(dict(id=row['id'],issue='independent_answer_mismatch',independent=answer,stored=row['ground_truth_answer']))
                if not check['evidence_matches']:
                    result['errors'].append(dict(id=row['id'],issue='independent_evidence_mismatch',missing=sorted(set(row['ground_truth_resource_ids'])-chart.evidence),extra=sorted(chart.evidence-set(row['ground_truth_resource_ids']))))
                cutoff=dt(row['oracle_spec']['as_of'])
                for ref in row['ground_truth_resource_ids']:
                    resource=snapshot.resources[ref]
                    owner=(resource.get('subject') or resource.get('patient') or {}).get('reference')
                    assert owner in (None,patient),'Evidence belongs to another patient'
                    for field in ('onsetDateTime','recordedDate','authoredOn','effectiveDateTime','performedDateTime','issued','abatementDateTime'):
                        if resource.get(field) and dt(resource[field])>=cutoff:
                            result['warnings']['evidence_field_at_or_after_cutoff'].append(dict(id=row['id'],resource=ref,field=field,time=resource[field]))
                    if resource['resourceType']=='Procedure' and resource.get('performedPeriod'):
                        check['procedure_period_representation']=True
                # Compare broader medication intent eligibility to the unstated intent=order restriction.
                non_orders=[r for r in snapshot.for_patient(patient,'MedicationRequest') if r.get('intent')!='order' and r.get('status') not in ('draft','entered-in-error') and dt(r['authoredOn'])<cutoff]
                if non_orders and row['family'] in ('inpatient_medications','admission_medication_comparison','followup_medication_comparison','medication_indications','preadmission_treated_conditions'):
                    result['warnings']['non_order_intent_candidates'].append(dict(id=row['id'],count=len(non_orders),intents=dict(Counter(r.get('intent') for r in non_orders))))
            except Exception as exc:
                result['errors'].append(dict(id=row['id'],issue=type(exc).__name__,detail=str(exc)))
            result['checks'].append(check)
        if i%20==0:print(f'Audited {i}/{len(grouped)} patients',flush=True)
    for family in sorted({r['family'] for r in rows}):
        group=[r for r in rows if r['family']==family]
        result['coverage'][family]=dict(total=len(group),splits=dict(Counter(r['split'] for r in group)))
    result['patient_splits']=dict(Counter(splits.values()))
    result['question_splits']=dict(Counter(r['split'] for r in rows))
    result['dataset_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==before[str(p.relative_to(ROOT))] for p in paths)
    result['independent_answer_matches']=sum(c.get('answer_matches',False) for c in result['checks'])
    result['independent_evidence_matches']=sum(c.get('evidence_matches',False) for c in result['checks'])
    result['passed_computational_checks']=not result['errors'] and not result['canonical_replay_errors'] and result['dataset_unchanged']
    OUT.mkdir(exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','warnings','errors','coverage')},indent=2))
    print('Errors:',len(result['errors']),'Warnings:',{k:len(v) for k,v in result['warnings'].items()})


if __name__=='__main__':main()
