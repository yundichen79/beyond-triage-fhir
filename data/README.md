# Clinical QA dataset

Detailed reference for the questions generated for Beyond TRIAGE-FHIR.
See the [project README](../README.md) for the experiment and pilot commands, the
[dataset guide](../docs/DATASET.md) for generation/verification procedures, and the
[current report](../docs/COHORT_200_REPORT.md) for validation status.

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

The examples below come from the **current training split**. They are selected
for readability, not as a representative performance sample. Every answer block
copies the complete stored `ground_truth_answer`; no patient values are invented.

To avoid repeating boilerplate, each displayed question omits its leading
`Patient <id>.` and the common suffix:

> Use records before 2026-10-02T00:00:00+00:00; use UTC times and only final,
> amended, or corrected observations.

The example ID locates its full question, patient reference, and evidence list in
`questions.jsonl`. Expand each family to inspect its rules and actual answer.

<details>
<summary><strong>Conditions recorded for an inpatient stay</strong> — <code>inpatient_conditions</code> (simple)</summary>

**Selection and meaning:** Select the most recent eligible completed inpatient encounter by its start time, then retrieve Conditions whose encounter reference points to it. Return coded concepts and onset dates. These are linked chart conditions, not an inferred principal diagnosis or reason for admission.

**Question** — `inpatient_conditions_0dbd62a19577fab41a1d` · training split

> For the most recent completed inpatient stay, list its linked condition codes, names, and recorded
> onset dates.

**Stored answer:**

```json
{
  "admitted_at": "2022-11-21T08:12:44+00:00",
  "conditions": [
    {
      "condition": {
        "code": "399211009",
        "name": "History of myocardial infarction (situation)",
        "system": "http://snomed.info/sct"
      },
      "onset": "2022-11-21T09:48:02+00:00"
    }
  ],
  "discharged_at": "2022-11-22T08:12:44+00:00"
}
```

**Retrieval work:** Encounter → linked Condition records. Stored support: 1 Condition, 1 Encounter.

</details>

<details>
<summary><strong>Medication orders linked to an inpatient stay</strong> — <code>inpatient_medications</code> (simple)</summary>

**Selection and meaning:** Choose the latest eligible inpatient stay with at least one qualifying order. Keep orders with intent=order, excluding draft and entered-in-error records, and resolve inline or referenced Medication codes. Separate orders for the same product remain separate answer entries.

**Question** — `inpatient_medications_105ebad64bd3136d038f` · training split

> For the most recent completed inpatient stay with at least one recorded non-draft, non-error
> medication order, list linked medication orders: coded product, authored date, and current recorded
> order status. Orders do not establish administration.

**Stored answer:**

```json
{
  "admitted_at": "2013-01-09T20:45:39+00:00",
  "discharged_at": "2013-01-12T04:34:39+00:00",
  "orders": [
    {
      "medication": {
        "code": "861467",
        "name": "Meperidine Hydrochloride 50 MG Oral Tablet",
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm"
      },
      "ordered_at": "2013-01-09T23:34:39+00:00",
      "recorded_status": "completed"
    }
  ]
}
```

**Retrieval work:** Encounter → MedicationRequest → Medication when referenced. Stored support: 1 Encounter, 1 MedicationRequest.

</details>

<details>
<summary><strong>Report-linked CBC results</strong> — <code>cbc_panel</code> (simple)</summary>

**Selection and meaning:** Choose the latest qualifying CBC DiagnosticReport (LOINC 58410-2). Follow its result references for hemoglobin (718-7) and platelets (777-3), retaining measurement times. An unrelated newer Observation cannot replace a report-linked result.

**Question** — `cbc_panel_05cbe8eec31181cd0fe5` · training split

> Review the latest final, amended, or corrected CBC report (LOINC 58410-2). Give its report time and
> the hemoglobin (718-7, g/dL) and platelet (777-3, 10*3/uL) results referenced by that report,
> including measurement times. Do not substitute unrelated observations.

**Stored answer:**

```json
{
  "hemoglobin": {
    "status": "found",
    "time": "2022-01-23T03:25:42+00:00",
    "unit": "g/dL",
    "value": "13.8"
  },
  "platelets": {
    "status": "found",
    "time": "2022-01-23T03:25:42+00:00",
    "unit": "10*3/uL",
    "value": "228.14"
  },
  "report_time": "2022-01-23T03:25:42+00:00"
}
```

**Retrieval work:** DiagnosticReport → result references → Observation. Stored support: 1 DiagnosticReport, 2 Observation.

</details>

<details>
<summary><strong>Blood pressure in the latest qualifying ambulatory visit</strong> — <code>outpatient_bp</code> (simple)</summary>

**Selection and meaning:** Choose the latest finished AMB encounter with a qualifying linked systolic observation. Extract systolic (8480-6) and diastolic (8462-4) from observation codes or components. Select each measurement at its latest eligible time within that encounter.

**Question** — `outpatient_bp_02247882a5d7727bf090` · training split

> For the most recent completed ambulatory encounter (class AMB) with a linked blood-pressure
> observation, report the most recent final, amended, or corrected systolic and diastolic
> blood-pressure measurements linked to that encounter, with values in mmHg and measurement times.
> Report a missing record or missing numeric value explicitly.

**Stored answer:**

```json
{
  "diastolic": {
    "status": "found",
    "time": "2026-01-24T03:35:40+00:00",
    "unit": "mm[Hg]",
    "value": "54"
  },
  "systolic": {
    "status": "found",
    "time": "2026-01-24T03:35:40+00:00",
    "unit": "mm[Hg]",
    "value": "95"
  },
  "visit_started_at": "2026-01-24T03:35:40+00:00"
}
```

**Retrieval work:** Encounter → linked Observation → component values. Stored support: 1 Encounter, 1 Observation.

</details>

<details>
<summary><strong>Documented prescribing history before admission</strong> — <code>preadmission_treated_conditions</code> (intermediate)</summary>

**Selection and meaning:** Use the latest completed inpatient stay. Include conditions with onset before admission and no recorded abatement before admission, supported by explicit Condition reasonReference links from qualifying orders in the preceding 365 days. Include the lower boundary; exclude admission itself. This establishes recorded prescribing history, not active disease or medication use.

**Question** — `preadmission_treated_conditions_36ae21be0a6be1dad9c9` · training split

> Review treatment history before the most recent completed inpatient stay. Which conditions had
> recorded onset before admission, no recorded abatement before admission, and an explicit Condition
> reasonReference from a non-draft, non-error medication order authored in the preceding 365 days,
> including the lower boundary and excluding admission itself? Give each condition's code, name, onset
> and abatement dates, and the linked coded medication products and order dates. Report no matching
> documented treatment history if none exists. These records do not establish medication use or active
> disease at admission.

**Stored answer:**

```json
{
  "admitted_at": "2021-09-13T22:48:30+00:00",
  "conditions_with_prior_orders": [
    {
      "abatement": null,
      "condition": {
        "code": "64859006",
        "name": "Osteoporosis (disorder)",
        "system": "http://snomed.info/sct"
      },
      "onset": "2015-07-12T21:42:28+00:00",
      "orders": [
        {
          "medication": {
            "code": "904419",
            "name": "Alendronic acid 10 MG Oral Tablet",
            "system": "http://www.nlm.nih.gov/research/umls/rxnorm"
          },
          "ordered_at": "2021-08-15T21:42:28+00:00"
        }
      ]
    }
  ],
  "discharged_at": "2021-09-18T23:03:30+00:00"
}
```

**Retrieval work:** Encounter → earlier MedicationRequest → reasonReference Condition; resolve medication. Stored support: 1 Condition, 1 Encounter, 1 MedicationRequest.

</details>

<details>
<summary><strong>Ambulatory follow-up after discharge</strong> — <code>discharge_followup</code> (intermediate)</summary>

**Selection and meaning:** Select the latest inpatient stay with a complete assigned follow-up window. Find the first finished AMB encounter strictly after discharge and no later than day 30, 60, or 90, as stated in the question. Return elapsed days and conditions linked to that visit, or followup=null.

**Question** — `discharge_followup_f976eb33059a7307e8ec` · training split

> For the most recent completed inpatient stay with a full 90-day window before the record cutoff,
> identify the first completed ambulatory encounter strictly after discharge and no later than 90 days
> afterward. Give its time, elapsed days rounded to one decimal, and linked conditions; otherwise
> report no matching follow-up.

**Stored answer:**

```json
{
  "admitted_at": "2015-10-06T15:18:31+00:00",
  "discharged_at": "2015-10-13T15:33:31+00:00",
  "followup": {
    "conditions": [
      {
        "condition": {
          "code": "65363002",
          "name": "Otitis media (disorder)",
          "system": "http://snomed.info/sct"
        },
        "onset": "2015-10-27T12:41:42+00:00"
      }
    ],
    "days_after_discharge": "13.9",
    "started_at": "2015-10-27T12:41:42+00:00"
  }
}
```

**Retrieval work:** Inpatient Encounter → bounded ambulatory Encounter search → linked Condition. Stored support: 1 Condition, 2 Encounter.

</details>

<details>
<summary><strong>Medication products across two inpatient stays</strong> — <code>admission_medication_comparison</code> (intermediate)</summary>

**Selection and meaning:** Select the two latest eligible inpatient stays and compare the sets of linked ordered products by coding system and product code. Return products in both stays, only the earlier stay, or only the later stay. There is no maximum gap between the two stays. Missing orders do not establish discontinuation.

**Question** — `admission_medication_comparison_fe356de1ecf69fb0c0df` · training split

> Compare coded medication products ordered during the two most recent completed inpatient stays.
> Report products in both, only the earlier stay, and only the later stay. Match by coding system and
> product code; absence does not establish discontinuation.

**Stored answer:**

```json
{
  "admitted_at": "2012-06-29T00:35:04+00:00",
  "discharged_at": "2012-07-04T00:50:04+00:00",
  "previous_admitted_at": "2012-06-26T18:54:04+00:00",
  "products": {
    "both": [],
    "only_first": [
      {
        "code": "861467",
        "name": "Meperidine Hydrochloride 50 MG Oral Tablet",
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm"
      }
    ],
    "only_second": []
  }
}
```

**Retrieval work:** Two Encounters → their MedicationRequests → medication code sets. Stored support: 2 Encounter, 1 MedicationRequest.

</details>

<details>
<summary><strong>Explicitly documented procedure indications</strong> — <code>procedure_indications</code> (intermediate)</summary>

**Selection and meaning:** Find the latest completed encounter with completed procedures having explicit Condition reasonReference links. Return those procedures, performed times, and their referenced condition indications. Same-encounter co-occurrence and procedure names do not establish an indication.

**Question** — `procedure_indications_a868741a4ada3be31572` · training split

> Find the most recent completed encounter containing completed procedures with explicit Condition
> reasonReference links. List those procedures with codes, names, performed times, and their linked
> condition indications. Exclude procedures without explicit Condition indication references; do not
> infer indications from procedure names or conditions elsewhere in the encounter.

**Stored answer:**

```json
{
  "encounter_started_at": "2026-01-30T04:03:23+00:00",
  "procedures": [
    {
      "indications": [
        {
          "code": "5602001",
          "name": "Opioid abuse",
          "system": "http://snomed.info/sct"
        }
      ],
      "performed_at": "2026-01-30T04:03:23+00:00",
      "procedure": {
        "code": "408919008",
        "name": "Psychosocial care (regime/therapy)",
        "system": "http://snomed.info/sct"
      }
    }
  ]
}
```

**Retrieval work:** Encounter → Procedure → reasonReference Condition. Stored support: 1 Condition, 1 Encounter, 1 Procedure.

</details>

<details>
<summary><strong>Explicitly documented prescribing indications</strong> — <code>medication_indications</code> (intermediate)</summary>

**Selection and meaning:** Find the latest completed encounter with qualifying orders having explicit Condition reasonReference links. Resolve the medication product and each referenced condition. Exclude orders without such indication references; do not infer an indication from a drug name.

**Question** — `medication_indications_08d7291612cdc91cb977` · training split

> Find the most recent completed encounter containing medication orders with explicit Condition
> reasonReference links. List those coded medications and their linked condition indications. Exclude
> orders without explicit indication references; do not infer indications from drug names.

**Stored answer:**

```json
{
  "encounter_started_at": "2026-08-27T11:56:46+00:00",
  "orders": [
    {
      "indications": [
        {
          "code": "55822004",
          "name": "Hyperlipidemia (disorder)",
          "system": "http://snomed.info/sct"
        }
      ],
      "medication": {
        "code": "314231",
        "name": "Simvastatin 10 MG Oral Tablet",
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm"
      }
    }
  ]
}
```

**Retrieval work:** Encounter → MedicationRequest → Condition and, when needed, Medication. Stored support: 1 Condition, 1 Encounter, 1 MedicationRequest.

</details>

<details>
<summary><strong>Short-interval CBC comparison</strong> — <code>cbc_change</code> (intermediate)</summary>

**Selection and meaning:** Find the latest qualifying CBC report with an earlier qualifying report in the preceding 90 days, including the lower boundary but excluding the same timestamp. Use the latest of those earlier reports. Follow each report’s own result references and calculate later-minus-earlier changes rounded to one decimal.

**Question** — `cbc_change_9c66ae9f7d1ae786c162` · training split

> Review a short-interval CBC trend. Find the most recent final, amended, or corrected CBC report
> (LOINC 58410-2) that has an earlier qualifying CBC report within the preceding 90 days, including
> the lower boundary and excluding the same timestamp. Compare it with the most recent of those
> earlier reports. For each, give referenced hemoglobin (718-7, g/dL) and platelet (777-3, 10*3/uL)
> results and dates, and the later-minus-earlier changes rounded to one decimal. Do not substitute
> reports farther apart or results from other reports; state any missing record or numeric value.

**Stored answer:**

```json
{
  "hemoglobin_change_g_dL": "3",
  "latest": {
    "hemoglobin": {
      "status": "found",
      "time": "2016-06-27T21:58:16+00:00",
      "unit": "g/dL",
      "value": "15.314"
    },
    "platelets": {
      "status": "found",
      "time": "2016-06-27T21:58:16+00:00",
      "unit": "10*3/uL",
      "value": "166.76"
    },
    "report_time": "2016-06-27T21:58:16+00:00"
  },
  "platelet_change_10_3_uL": "7",
  "previous": {
    "hemoglobin": {
      "status": "found",
      "time": "2016-06-27T09:17:14+00:00",
      "unit": "g/dL",
      "value": "12.364"
    },
    "platelets": {
      "status": "found",
      "time": "2016-06-27T09:17:14+00:00",
      "unit": "10*3/uL",
      "value": "159.72"
    },
    "report_time": "2016-06-27T09:17:14+00:00"
  }
}
```

**Retrieval work:** Select two DiagnosticReports → traverse each result set → compare values. Stored support: 2 DiagnosticReport, 4 Observation.

</details>

<details>
<summary><strong>First-year HbA1c testing after recorded glycemic onset</strong> — <code>glycemic_followup</code> (complex)</summary>

**Selection and meaning:** Anchor on the earliest recorded onset of type 2 diabetes (SNOMED 44054006) or prediabetes (714628002). Find the first eligible HbA1c (LOINC 4548-4) strictly after onset and before day 365. If present, compare with the latest eligible baseline in the 90 days up to and including onset. Require a full follow-up window; keep missing baseline and missing follow-up distinct.

**Question** — `glycemic_followup_3dc9d97fa0dd08d70287` · training split

> Review documented first-year glycemic testing after the first recorded onset of type 2 diabetes
> (SNOMED 44054006) or prediabetes (714628002). Was HbA1c (LOINC 4548-4) measured strictly afterward
> and before day 365? If so, give the first result and compare it with the latest result in the 90
> days up to and including onset. Include dates, values in percent, and change in percentage points
> rounded to one decimal. Require the full follow-up window before the record cutoff. Distinguish
> missing follow-up, missing baseline, and missing numeric values; do not infer treatment effect or
> whether care met a guideline.

**Stored answer:**

```json
{
  "baseline": {
    "status": "found",
    "time": "2011-11-05T03:35:40+00:00",
    "unit": "%",
    "value": "5.93"
  },
  "change_percentage_points": "0.1",
  "condition": {
    "code": "714628002",
    "name": "Prediabetes (finding)",
    "system": "http://snomed.info/sct"
  },
  "followup": {
    "status": "found",
    "time": "2012-05-12T03:35:40+00:00",
    "unit": "%",
    "value": "6.01"
  },
  "onset": "2011-11-05T03:35:40+00:00"
}
```

**Retrieval work:** Condition onset → bounded follow-up Observation search → conditional baseline search → difference. Stored support: 1 Condition, 2 Observation.

</details>

<details>
<summary><strong>First-year creatinine follow-up</strong> — <code>renal_followup</code> (complex)</summary>

**Selection and meaning:** Anchor on the first eligible creatinine measurement (LOINC 2160-0) linked to a completed AMB encounter. Find the first measurement strictly afterward and before day 365, then calculate follow-up minus baseline in mg/dL. Require the full follow-up window before the cutoff.

**Question** — `renal_followup_f72f461fc08eea24ebb4` · training split

> Review documented first-year renal laboratory follow-up, starting with the first recorded final,
> amended, or corrected creatinine measurement (LOINC 2160-0) linked to a completed ambulatory
> encounter. Was creatinine remeasured strictly afterward and before day 365? If so, compare the first
> subsequent result with that baseline. Give dates, values in mg/dL, and change rounded to one
> decimal. Require the full follow-up window before the record cutoff, and distinguish no follow-up
> from missing numeric values. This is a record review, not a judgment about recommended monitoring
> intervals.

**Stored answer:**

```json
{
  "baseline": {
    "status": "found",
    "time": "2015-05-12T07:52:18+00:00",
    "unit": "mg/dL",
    "value": "2.6071"
  },
  "change_mg_dL": "-1",
  "followup": {
    "status": "found",
    "time": "2015-06-05T06:44:05+00:00",
    "unit": "mg/dL",
    "value": "1.6355"
  }
}
```

**Retrieval work:** Observation → linked ambulatory Encounter → follow-up Observation search → difference. Stored support: 1 Encounter, 2 Observation.

</details>

<details>
<summary><strong>Ordered products at inpatient and ambulatory encounters</strong> — <code>followup_medication_comparison</code> (complex)</summary>

**Selection and meaning:** Choose the latest eligible inpatient stay with qualifying orders and a complete 30-, 60-, or 90-day window. Find the first qualifying ambulatory follow-up and compare product-code sets across the two encounters. If a follow-up exists, the inpatient set is nonempty by construction; otherwise return followup=null. These are encounter orders, not discharge prescriptions or reconciled medication use.

**Question** — `followup_medication_comparison_14452e174389783b84e3` · training split

> For the most recent completed inpatient stay that contains at least one recorded non-draft,
> non-error medication order and has a full 60-day window before the record cutoff, find the first
> completed ambulatory encounter strictly after discharge and no later than day 60. If one exists,
> compare coded medication products ordered in the stay and at follow-up: both, inpatient only, and
> follow-up only. Match by coding system and product code. Otherwise report no matching follow-up.
> Compare encounter-linked orders; these are not discharge prescriptions or a reconciled list of
> medications being taken, and missing orders do not establish discontinuation.

**Stored answer:**

```json
{
  "admitted_at": "2023-09-05T19:18:50+00:00",
  "discharged_at": "2023-09-08T20:28:09+00:00",
  "followup": {
    "days_after_discharge": "4",
    "started_at": "2023-09-12T20:28:09+00:00"
  },
  "products": {
    "both": [],
    "only_first": [
      {
        "code": "1719286",
        "name": "10 ML Furosemide 10 MG/ML Injection",
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm"
      }
    ],
    "only_second": []
  }
}
```

**Retrieval work:** Inpatient Encounter/orders → conditional follow-up Encounter search → orders → set comparison. Stored support: 2 Encounter, 6 Medication, 6 MedicationRequest.

</details>

<details>
<summary><strong>Subsequent admission and shared recorded conditions</strong> — <code>subsequent_admission</code> (complex)</summary>

**Selection and meaning:** Start with the first recorded eligible completed inpatient stay. Find the next completed inpatient admission strictly after its discharge and before the cutoff. Return its admission time, elapsed days, linked conditions, and condition codes shared with the initial stay. This question has no fixed 30-day readmission window.

**Question** — `subsequent_admission_f6a82c26597dd19abc2c` · training split

> After the first recorded completed inpatient stay, identify the next completed inpatient admission
> strictly after its discharge and before the record cutoff. If one exists, give its admission time,
> the elapsed days rounded to one decimal, its linked conditions, and condition codes shared by the
> two stays. Otherwise report no subsequent recorded admission.

**Stored answer:**

```json
{
  "admitted_at": "1973-02-20T01:53:54+00:00",
  "discharged_at": "1973-03-18T02:08:54+00:00",
  "subsequent_admission": {
    "admitted_at": "2004-08-14T02:20:31+00:00",
    "conditions": [
      {
        "condition": {
          "code": "399211009",
          "name": "History of myocardial infarction (situation)",
          "system": "http://snomed.info/sct"
        },
        "onset": "2004-08-14T20:21:49+00:00"
      }
    ],
    "days_after_discharge": "11472",
    "shared_conditions": []
  }
}
```

**Retrieval work:** First inpatient Encounter → next eligible Encounter → both condition sets → intersection. Stored support: 1 Condition, 2 Encounter.

</details>

## A real negative example

A missing follow-up is still answerable from the chart. This training example has
an eligible inpatient anchor but no qualifying ambulatory follow-up in its stated
window. The stored answer uses `null`, not an invented visit or a query-error label.

Example: `discharge_followup_1fe9dd2ddbcf1a58f72e`

> For the most recent completed inpatient stay with a full 90-day window before the record cutoff,
> identify the first completed ambulatory encounter strictly after discharge and no later than 90 days
> afterward. Give its time, elapsed days rounded to one decimal, and linked conditions; otherwise
> report no matching follow-up.

```json
{
  "admitted_at": "2026-02-05T01:21:08+00:00",
  "discharged_at": "2026-02-06T01:21:08+00:00",
  "followup": null
}
```

The same patient could have unrelated encounters or follow-up outside this
window. “No matching follow-up” is scoped to the question, not a claim about all care.

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
remain separate checks. See the [current report](../docs/COHORT_200_REPORT.md).

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
