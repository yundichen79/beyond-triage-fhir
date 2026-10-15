# Current clinical QA cohort

The current dataset contains 1,202 questions from 200 preserved Synthea patients
across 14 clinical chart-review families. The generator and answer rules were
refined and the current dataset regenerated in place.

| Split | Patients | Questions |
|---|---:|---:|
| Train | 151 | 899 |
| Validation | 22 | 145 |
| Test | 27 | 158 |

Patient assignments are unchanged. All 200 patients have questions. Every family
appears in all three splits. Difficulty labels are provisional: 499 simple,
548 intermediate, and 155 complex examples. They describe intended operations,
not measured model performance.

| Family | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `admission_medication_comparison` | 15 | 4 | 2 | 21 |
| `cbc_change` | 28 | 4 | 7 | 39 |
| `cbc_panel` | 151 | 22 | 27 | 200 |
| `discharge_followup` | 55 | 11 | 8 | 74 |
| `followup_medication_comparison` | 15 | 3 | 2 | 20 |
| `glycemic_followup` | 12 | 1 | 1 | 14 |
| `inpatient_conditions` | 53 | 11 | 10 | 74 |
| `inpatient_medications` | 19 | 4 | 2 | 25 |
| `medication_indications` | 137 | 21 | 26 | 184 |
| `outpatient_bp` | 151 | 22 | 27 | 200 |
| `preadmission_treated_conditions` | 32 | 7 | 5 | 44 |
| `procedure_indications` | 139 | 21 | 26 | 186 |
| `renal_followup` | 17 | 2 | 6 | 25 |
| `subsequent_admission` | 75 | 12 | 9 | 96 |

The generator recomputed and validated all 1,202 answers and evidence sets using
a freshly built temporary index before replacing the current dataset. The
preserved source cohort contains 528,053 normalized resources; raw records and
their checksums were not changed. All 1,202 questions also passed full-cohort
source replay against the original checksummed bundles. Replay uses the same
answer rules, not an independently implemented oracle. All 74 fixture and
regression tests pass, covering clinical selection and time boundaries, live-server
lifecycle and import probes, and the zero-shot agent's tools, scoring and limits.
Scripted agent tests verify software behavior; they do not measure Qwen accuracy.

There are 90 sampled primary negative cases. This counts missing follow-up or
empty requested record sets according to each family definition; it is not a
measure of every missing field. Sixty-seven questions include evidence from
separate Medication resources; CBC families also require report-result traversal.

The refined families have tighter clinical scope:

- Procedure indications require explicit Condition reason references. Same-encounter
  co-occurrence cannot establish a procedure's indication.
- Preadmission treatment history requires explicit prescribing evidence in the
  preceding 365 days for a condition with onset before admission and no abatement
  before admission. There are 33 positive and 11 sampled empty histories.
- Medication follow-up starts from an inpatient stay with qualifying orders.
  There are 15 documented follow-ups with a nonempty inpatient product set and
  five sampled cases without follow-up. No returned comparison has both product
  sets empty. These are encounter-order comparisons, not discharge medication lists.
- CBC trends require reports at most 90 days apart. The latest qualifying pair
  is selected; charts without such a pair do not receive this question.
- Glycemic and renal questions review documented testing in the first 365 days
  after their respective anchors. These explicit task parameters do not assess
  guideline adherence or treatment effects. Discharge windows remain 30/60/90 days.

Small families remain pilot-scale: glycemic follow-up has one validation and
one test example, renal follow-up has two validation examples, and inpatient
medication comparison has two test examples. These counts cannot support precise
per-family accuracy estimates. More patients are needed for reliable family-level
evaluation. Agent pilots must still check for shortcuts, vocabulary coverage,
appropriate difficulty, and the distribution of actual tool effort.

The user confirmed that clinical review passed for the current 1,202-question
dataset. Live job 29686963 checked this version, but failed the large-chart
preflight: 45,636 resources were imported, then patient-scoped searches returned
none of the expected 35,473 clinical records. Full-cohort import did not begin.
Servers were stopped and temporary artifacts cleaned up. A HAPI configuration
workaround and an early imported-record search probe are ready; a successful live
rerun is still required.

The zero-shot Qwen3-8B pilot is implemented. Its deterministic training-only
selection contains 56 questions from 44 patients, four per family, including eight
primary negative cases. Inputs exclude ground truth and question metadata;
answers, retrieved evidence, tool effort and tokens are scored outside the model.
The launcher verifies the selected charts before model execution and keeps one
result file, `outputs/pilot.jsonl`. Model preparation and an actual GPU run remain
pending. No real-model accuracy or SFT trajectories have been collected or claimed.

The processed directory contains only `questions.jsonl` and
`patient_splits.json`. Coverage and check results print to the terminal; see
the [dataset guide](DATASET.md) for rebuilding and verification commands.
