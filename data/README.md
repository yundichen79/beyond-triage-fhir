# Clinical QA dataset

Detailed reference for the questions generated for Beyond TRIAGE-FHIR.
See the [project README](../README.md) for the experiment, dataset construction,
live verification, and pilot commands.

## Cohort and files

The current cohort contains **200 synthetic Synthea v4.0.0 patients**, generated
for Massachusetts with complete exported FHIR R4 histories. Questions use the
exclusive record cutoff **2026-10-02T00:00:00+00:00**. The raw generation record
preserves seeds, configuration, generator identity, and source-file checksums.

```text
data/
├── README.md
├── raw/batch-001/
│   ├── generation.json
│   └── fhir/                    Original patient and provider bundles
└── processed/cohort-200/
    ├── questions.jsonl          One question and its labels per line
    └── patient_splits.json      Patient reference → train / validation / test
```

Only the two dataset files above are kept in the processed directory. Normalized
indexes and live FHIR databases are temporary. Raw and processed data are ignored
by Git; this README is kept separately from the generated files.

There are **1,202 questions across 14 families**. Every patient has questions,
and every family occurs in all three splits.

| Split | Patients | Questions |
| --- | ---: | ---: |
| Train | 151 | 899 |
| Validation | 22 | 145 |
| Test | 27 | 158 |

Patient assignments use a stable hash and the configured 70/15/15 target ratios;
the realized counts above need not match those proportions exactly. All questions
for one patient share its split. Assignments remain stable when patients are added.

## What the questions test

Questions are generated from **14 explicit templates with deterministic answer
rules**, not sampled from an LLM. Each family defines eligible clinical anchors,
resource relationships, temporal boundaries, and the required answer structure.
At most one question per patient per family is retained. A chart without the
required anchor does not receive that family’s question.

The intended progression is:

- **Simple:** select the right encounter or report, then retrieve its linked records.
- **Intermediate:** combine temporal selection, explicit indication references,
  report comparisons, or medication-set operations.
- **Complex:** anchor a longitudinal review, conditionally search for follow-up,
  and compare records across time or encounters.

These labels describe intended operations, not measured model difficulty. A
negative case may terminate earlier than a positive case in the same family.

| Family | Difficulty | Questions | Primary negatives |
| --- | --- | ---: | ---: |
| `inpatient_conditions` | simple | 74 | 18 |
| `inpatient_medications` | simple | 25 | 0 |
| `cbc_panel` | simple | 200 | 0 |
| `outpatient_bp` | simple | 200 | 0 |
| `preadmission_treated_conditions` | intermediate | 44 | 11 |
| `discharge_followup` | intermediate | 74 | 18 |
| `admission_medication_comparison` | intermediate | 21 | 5 |
| `procedure_indications` | intermediate | 186 | 0 |
| `medication_indications` | intermediate | 184 | 0 |
| `cbc_change` | intermediate | 39 | 0 |
| `glycemic_followup` | complex | 14 | 3 |
| `renal_followup` | complex | 25 | 6 |
| `followup_medication_comparison` | complex | 20 | 5 |
| `subsequent_admission` | complex | 96 | 24 |

Totals: **499 simple, 548 intermediate, 155 complex; 90 primary negatives**.
Primary negatives are family-defined empty histories, empty product/condition sets,
or absent follow-up records. They do not count every missing field in an answer.
The generator retains at most `max(1, positive_count // 3)` negatives per family
when positives exist, using stable hash ordering; families with no positives
receive no negative-only examples.

## Shared interpretation rules

- **Clinical scope:** questions describe documented records. Encounter-linked
  conditions are not automatically admission diagnoses; orders do not establish
  administration, actual use, or discontinuation. Indications require explicit
  Condition references, not co-occurrence or medical inference.
- **Time:** timestamps use UTC. Inpatient anchors are finished IMP encounters with
  valid periods ending before the cutoff; ambulatory encounters use class AMB.
  Earliest/latest encounter and report anchors are selected by their start or
  effective time. Ambiguous anchor timestamp ties are rejected.
- **Observations:** use final, amended, or corrected records. Measurements may be
  stored directly or as components. CBC answers follow DiagnosticReport.result
  references; observation time comes from effectiveDateTime/effectivePeriod.
- **Medication identity:** use coding system and product code. Orders may contain
  inline medication codes or references to Medication resources. Product
  comparisons deduplicate products; order-list questions retain separate orders.
- **Windows:** discharge follow-up uses an explicitly stated 30, 60, or 90 days,
  assigned deterministically by patient and family. The window includes its final
  day and excludes discharge itself. First-year laboratory follow-up excludes
  day 365. Anchors must have a complete assigned follow-up window before the cutoff.
- **Calculations:** changes are later minus earlier. Requested changes and elapsed
  days are rounded to one decimal using decimal half-even rounding; trailing zeros
  can be removed. Measurement precision is retained. Numeric results are stored as
  strings; a rounded `"0"` or `"-0"` is a valid zero, not missing data.
- **Invalid data:** unresolved references, cross-patient references, unsupported
  quantities, or conflicting values at a selected measurement time are rejected
  rather than converted into missing-record answers. Equal-time measurements with
  the same value can contribute multiple supporting IDs.

The follow-up intervals are task definitions for record review. They do not
establish guideline adherence, treatment effects, or recommended monitoring frequency.

## Question fields

Each `questions.jsonl` line is one JSON object with these 11 fields:

| Field | Type | Meaning | Sent to answering model? |
| --- | --- | --- | --- |
| `id` | string | Stable identifier derived from family, patient, and computation specification | No |
| `patient_id` | string | FHIR patient reference, such as `Patient/<logical-id>` | Yes |
| `question` | string | Full request, including patient context, cutoff, and applicable task rules | Yes |
| `family` | string | One of the 14 categories above | No |
| `complexity` | string | Provisional simple/intermediate/complex label | No |
| `required_operations` | string array | Intended retrieval/comparison operations for analysis | No |
| `split` | string | Patient’s train/validation/test assignment | No |
| `ground_truth_answer` | object | Deterministically computed structured answer | No |
| `ground_truth_resource_ids` | string array | Supporting FHIR resource references | No |
| `answer_type` | string | Currently `object` for every question | No |
| `oracle_spec` | object | Evaluator operation, cutoff, and family-specific window parameters | No |

The model also receives shared tool definitions and a public answer schema.
Evaluator labels and answer-specific resource IDs stay outside the prompt.
Supporting IDs identify answer-associated records; they are not an exhaustive
certificate that an absence claim or earliest/latest selection was justified.
The agent searches complete patient charts, including uncited records.

### Missing data and empty answers

| Representation | Interpretation |
| --- | --- |
| Measurement with `status: "found"`, value, time, and unit | An eligible numeric result exists |
| Measurement with `status: "no_matching_record"` | No eligible matching measurement was found |
| Measurement with `status: "missing_value"`, `value: null`, time, and unit | A matching record exists without a numeric quantity |
| `followup: null` or `subsequent_admission: null` | No matching encounter in that question’s search scope |
| An empty conditions/orders/product array | An eligible anchor exists, but that requested set is empty |
| A calculated change of `null` | The necessary numeric pair is unavailable |

The current answers contain **no `missing_value` measurement examples**, although
the format and evaluator support them. Empty results are valid labels; HTTP errors
or unfinished searches are not evidence of absence.

## Real question examples

These questions come from the current training split. Patient IDs and the repeated
record cutoff/status instructions are omitted for readability. The wording below
is copied from the generated questions.

### Simple

**Conditions recorded for an inpatient stay**

> For the most recent completed inpatient stay, list its linked condition codes, names, and recorded
> onset dates.

**Medication orders linked to an inpatient stay**

> For the most recent completed inpatient stay with at least one recorded non-draft, non-error
> medication order, list linked medication orders: coded product, authored date, and current recorded
> order status. Orders do not establish administration.

**Report-linked CBC results**

> Review the latest final, amended, or corrected CBC report (LOINC 58410-2). Give its report time and
> the hemoglobin (718-7, g/dL) and platelet (777-3, 10*3/uL) results referenced by that report,
> including measurement times. Do not substitute unrelated observations.

**Blood pressure in the latest qualifying ambulatory visit**

> For the most recent completed ambulatory encounter (class AMB) with a linked blood-pressure
> observation, report the most recent final, amended, or corrected systolic and diastolic
> blood-pressure measurements linked to that encounter, with values in mmHg and measurement times.
> Report a missing record or missing numeric value explicitly.

### Intermediate

**Documented prescribing history before admission**

> Review treatment history before the most recent completed inpatient stay. Which conditions had
> recorded onset before admission, no recorded abatement before admission, and an explicit Condition
> reasonReference from a non-draft, non-error medication order authored in the preceding 365 days,
> including the lower boundary and excluding admission itself? Give each condition's code, name, onset
> and abatement dates, and the linked coded medication products and order dates. Report no matching
> documented treatment history if none exists. These records do not establish medication use or active
> disease at admission.

**Ambulatory follow-up after discharge**

> For the most recent completed inpatient stay with a full 90-day window before the record cutoff,
> identify the first completed ambulatory encounter strictly after discharge and no later than 90 days
> afterward. Give its time, elapsed days rounded to one decimal, and linked conditions; otherwise
> report no matching follow-up.

**Medication products across two inpatient stays**

> Compare coded medication products ordered during the two most recent completed inpatient stays.
> Report products in both, only the earlier stay, and only the later stay. Match by coding system and
> product code; absence does not establish discontinuation.

**Explicitly documented procedure indications**

> Find the most recent completed encounter containing completed procedures with explicit Condition
> reasonReference links. List those procedures with codes, names, performed times, and their linked
> condition indications. Exclude procedures without explicit Condition indication references; do not
> infer indications from procedure names or conditions elsewhere in the encounter.

**Explicitly documented prescribing indications**

> Find the most recent completed encounter containing medication orders with explicit Condition
> reasonReference links. List those coded medications and their linked condition indications. Exclude
> orders without explicit indication references; do not infer indications from drug names.

**Short-interval CBC comparison**

> Review a short-interval CBC trend. Find the most recent final, amended, or corrected CBC report
> (LOINC 58410-2) that has an earlier qualifying CBC report within the preceding 90 days, including
> the lower boundary and excluding the same timestamp. Compare it with the most recent of those
> earlier reports. For each, give referenced hemoglobin (718-7, g/dL) and platelet (777-3, 10*3/uL)
> results and dates, and the later-minus-earlier changes rounded to one decimal. Do not substitute
> reports farther apart or results from other reports; state any missing record or numeric value.

### Complex

**First-year HbA1c testing after recorded glycemic onset**

> Review documented first-year glycemic testing after the first recorded onset of type 2 diabetes
> (SNOMED 44054006) or prediabetes (714628002). Was HbA1c (LOINC 4548-4) measured strictly afterward
> and before day 365? If so, give the first result and compare it with the latest result in the 90
> days up to and including onset. Include dates, values in percent, and change in percentage points
> rounded to one decimal. Require the full follow-up window before the record cutoff. Distinguish
> missing follow-up, missing baseline, and missing numeric values; do not infer treatment effect or
> whether care met a guideline.

**First-year creatinine follow-up**

> Review documented first-year renal laboratory follow-up, starting with the first recorded final,
> amended, or corrected creatinine measurement (LOINC 2160-0) linked to a completed ambulatory
> encounter. Was creatinine remeasured strictly afterward and before day 365? If so, compare the first
> subsequent result with that baseline. Give dates, values in mg/dL, and change rounded to one
> decimal. Require the full follow-up window before the record cutoff, and distinguish no follow-up
> from missing numeric values. This is a record review, not a judgment about recommended monitoring
> intervals.

**Ordered products at inpatient and ambulatory encounters**

> For the most recent completed inpatient stay that contains at least one recorded non-draft,
> non-error medication order and has a full 60-day window before the record cutoff, find the first
> completed ambulatory encounter strictly after discharge and no later than day 60. If one exists,
> compare coded medication products ordered in the stay and at follow-up: both, inpatient only, and
> follow-up only. Match by coding system and product code. Otherwise report no matching follow-up.
> Compare encounter-linked orders; these are not discharge prescriptions or a reconciled list of
> medications being taken, and missing orders do not establish discontinuation.

**Subsequent admission and shared recorded conditions**

> After the first recorded completed inpatient stay, identify the next completed inpatient admission
> strictly after its discharge and before the record cutoff. If one exists, give its admission time,
> the elapsed days rounded to one decimal, its linked conditions, and condition codes shared by the
> two stays. Otherwise report no subsequent recorded admission.

## Validation and agent use

Question templates and generation are implemented in
[dataset/questions.py](../dataset/questions.py); deterministic answers are defined
in [dataset/oracle.py](../dataset/oracle.py). Run from the repository root:

```bash
python3 -m dataset.validate --dataset data/processed/cohort-200
python3 -m dataset.review --dataset data/processed/cohort-200 --bundles data/raw/batch-001/fhir
```

Local validation, raw-bundle replay, and clinical review passed for this cohort.
Replay uses the same oracle; live FHIR verification and actual model execution
remain separate checks; current run status is described in the project README.

The zero-shot pilot selects 56 training questions from 44 patients, four per family,
including eight primary negatives. Agent episodes, scores, and token/tool metrics
go to `outputs/pilot.jsonl`, outside this dataset. QA rows are not SFT trajectories:
SFT requires actual tool interactions, checked answers, adequate retrieved/cited
evidence, and review of unfinished searches and selection logic.

Answer scoring normalizes coded identity, timestamp equivalence, supported unit
aliases, and unordered lists while preserving multiplicity. Numeric tolerance is
0.000001 in the requested units. Answer correctness and required-evidence retrieval/
citation coverage are reported separately; full evidence coverage alone cannot
prove a justified negative answer.

## Current coverage limits

This is a synthetic, template-based chart-review dataset. Patient records, dates,
codes, follow-up availability, and required operations vary, but the current release
has only 14 question templates; it does not establish broad free-form clinical QA
coverage or performance on real EHRs. Primary negative sampling also changes the
natural distribution of missing results.

Some families are small: glycemic follow-up has only 14 examples, with one each
in validation and test. Per-family performance estimates and the intended
complexity levels need larger cohorts and measured agent runs. Every stored row
has a deterministic answer, including explicit absence where applicable; this does
not yet establish that the agent can retrieve or produce it correctly.
