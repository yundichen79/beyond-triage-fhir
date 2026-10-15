# Clinical question dataset

Run commands from the repository root. The processed dataset has exactly two files:

| File | Purpose |
|---|---|
| `data/processed/cohort-200/questions.jsonl` | One clinical question per line, including structured ground truth and supporting resource references. |
| `data/processed/cohort-200/patient_splits.json` | Patient IDs mapped to train, validation, or test. No other metadata. |

Each question has 11 fields: `id`, `patient_id`, `question`, `family`, `complexity`,
`required_operations`, `split`, `ground_truth_answer`, `ground_truth_resource_ids`,
`answer_type`, and `oracle_spec`. Answers are JSON objects, with arrays for lists of
conditions or medications. `oracle_spec` is used by the evaluator, never the agent.
There are no separate split exports, input copies, review queues, or check reports.

The 14 current families cover inpatient conditions and medication orders,
encounter-linked outpatient blood pressure, recent preadmission treatment history,
CBC report results and short-interval changes, explicit procedure and prescribing
indications, discharge follow-up and encounter-linked medication comparisons,
first-year glycemic and renal lab follow-up, and subsequent inpatient admissions. Difficulty labels are
provisional pending agent pilots; larger answers alone do not establish difficulty.

Temporal boundaries are explicit in each question. Discharge follow-up windows
vary between 30, 60, and 90 days; glycemic and renal follow-up use the first 365
days, excluding the upper boundary. These are record-review task parameters,
not recommendations about testing frequency or judgments of guideline adherence.
Index events must have a complete
observation window before the frozen record cutoff. Clinical questions distinguish
orders from administrations, missing records from missing numeric values, and
recorded timelines from claims about treatment effects. Coded medication products
are compared by system and code; different products do not automatically imply a
change of ingredient, dose, or treatment plan.

The refined families apply the following eligibility rules:

- `procedure_indications` follows explicit Procedure-to-Condition reason references.
  Conditions or medications that merely share an encounter cannot establish an indication.
- `preadmission_treated_conditions` requires a pre-existing condition without
  abatement before admission and an explicit medication indication reference from
  an order authored in the preceding 365 days. The lower boundary is included;
  admission itself is excluded. Untreated conditions are outside this question's
  scope. This is documented prescribing history, not proof of active disease or use.
- `followup_medication_comparison` starts from the most recent eligible inpatient
  stay with at least one qualifying order. Its inpatient product set is therefore
  nonempty whenever a follow-up comparison is returned. No follow-up remains a
  valid negative. Encounter orders are not a discharge medication list.
- `cbc_change` finds the latest CBC with an earlier qualifying report within
  90 days, then chooses the latest such earlier report. Both results must follow
  their own report references. A chart with only reports farther apart is ineligible;
  the generator does not substitute a distant pair.

Questions require an eligible anchor. True negatives are sampled by a stable hash
within each family, using at most one negative per three positives (one is permitted
for a sparse positive family). Families with no positive examples are not filled
with empty questions. Missing follow-up can terminate an answer early. Structured
answers and supporting resources are checked deterministically before replacement.

Use the safe input loader:

```python
from pathlib import Path
from dataset.common import iter_agent_inputs

for example in iter_agent_inputs(Path("data/processed/cohort-200"), split="test"):
    # Only id, question, patient_id, and split are returned.
    # Supply question and patient context to the answering agent.
    print(example["question"])
```

## Build and validate

The original raw records are preserved. To rebuild the current QA:

```bash
python3 -m dataset.questions --output data/processed/cohort-200 --replace
python3 -m dataset.validate --dataset data/processed/cohort-200
python3 -m dataset.review --dataset data/processed/cohort-200 --bundles data/raw/batch-001/fhir
```

The generator prints coverage and exclusions to the terminal. It validates all
answers before writing. `--replace` updates the two files in the current directory;
it refuses a directory containing unrelated files. Without that flag an existing
dataset is protected. The validator checks canonical text/rules, question IDs,
patient splits, answers, and supporting references. Source replay reloads original
checksummed bundles and applies the same answer rules. This checks source agreement
but is not an independent implementation or a substitute for human clinical review.
Neither validation command changes or adds dataset files.

`configs/cohort.json` contains generation settings. The split seed
`beyond-triage-v1` is unchanged to preserve patient assignments. The frozen Synthea
end date is 2026-10-01; questions use the exclusive cutoff 2026-10-02 UTC.

For a fresh workspace, generate the Synthea records first:

```bash
python3 -m dataset.cohort download
sbatch scripts/generate_cohort.sbatch 1
```

Original Synthea bundles and their generation record remain in `data/raw/batch-001`.
Generation, validation, and live import read those bundles using a temporary
SQLite index. The index is normalized, checked for missing references and duplicate
patients, then removed when the command finishes. No permanent snapshot or second
resource export is kept. Rebuilding the index adds runtime on each invocation.
Use `--bundles PATH [PATH ...]` to select other batches. An explicitly supplied
external normalized source can still be read with `--snapshot PATH`.

## Live verification

```bash
python3 -m dataset.live prepare
sbatch scripts/verify_live_fhir.sbatch
```

The job loads Java 21 and PostgreSQL 16.4, starts isolated servers, imports complete
charts and their referenced resources, checks the largest chart for pagination,
and verifies answers from HTTP-retrieved resources. It includes DiagnosticReport
and resolves external Medication references. A search probe runs after the first
clinical chunk; unsearchable imports fail with diagnostics in stdout.
The current HAPI configuration uses `search_prefetch_thresholds: [100000, -1]`
and an unlimited fallback match-ID fetch limit. Clients explicitly request bounded
pages and check accurate totals. This configuration change follows a failed run
with empty patient searches and still needs live confirmation.

The source index, server database, and operational job artifacts use temporary
storage. Results and error excerpts print to stdout, which Slurm captures in
`logs/`; temporary artifacts are removed after completion. To intentionally retain
a particular run, explicitly supply `--output PATH` (the path must not exist).
The prepared vendor artifact contains only the pinned HAPI WAR and its identity.
Live jobs write no QA metadata. The new families still require a fresh live run
and agent pilots.

Clinical review has passed for the current cohort. Next, verify these QA tasks
against live FHIR and run an agent pilot, then collect actual ReAct tool
interactions using the final tool interface, retain efficient correct
trajectories with adequate evidence, then run SFT. Patient splits are shared across
all questions; labels, family, operation hints and computation rules stay out of
agent inputs.
