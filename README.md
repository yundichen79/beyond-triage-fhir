# Beyond TRIAGE-FHIR

Adaptive question answering over FHIR electronic health records, extending
[TRIAGE-FHIR](https://github.com/yundichen79/triage-fhir).

The planned comparison evaluates a fixed workflow, a zero-shot Qwen3-8B ReAct agent,
and an agent trained with trajectory supervised fine-tuning (SFT). We measure
answer accuracy alongside tool calls and token usage.

See the [question catalog](data/README.md) for dataset rules, fields, and examples.

## Current status

| Component | Status |
| --- | --- |
| Dataset | Built; local validation, source replay, and clinical review passed |
| Live FHIR verification | Passed for 200 patients / 1,202 questions; v4 query smoke checks passed 18/18 |
| Zero-shot pilot | V4 training diagnostic: 3/14 correct, 2/14 correct with required evidence; v5 implemented, model evaluation pending |
| SFT and comparative evaluation | Planned; fixed-workflow comparison, SFT, and held-out evaluation outstanding |

## Data source and dataset

**200 synthetic Massachusetts patients** were generated with Synthea v4.0.0 as
FHIR R4 bundles with complete histories. Seeds and dates are pinned in
[cohort.json](configs/cohort.json); exporter settings are in
[synthea.properties](configs/synthea.properties). Original bundles and their
generation record with checksums are preserved in `data/raw/batch-001/`.

The dataset contains **1,202 questions across 14 clinical chart-review families**,
using an exclusive cutoff of **2026-10-02 UTC**. Tasks cover encounter-linked
conditions/orders, CBC and blood pressure, documented indications, medication
comparisons, and longitudinal follow-up. Difficulty is provisional:
499 simple, 548 intermediate, and 155 complex.

| Split | Patients | Questions |
| --- | ---: | ---: |
| Train | 151 | 899 |
| Validation | 22 | 145 |
| Test | 27 | 158 |

All questions for a patient stay in the same split.

## Dataset construction

1. Preserve Synthea bundles and normalize references in a temporary SQLite index.
2. Generate eligible questions and deterministic Python answers, with explicit time
   windows, units, supporting references, and sampled missing-record cases.
3. Validate answers, evidence, question rules, and patient splits.
4. Replay answers from raw bundles and conduct clinical review. Replay uses the
   same answer rules; clinical review provides a separate check.

Only two processed files are kept under `data/processed/cohort-200/`:

| File | Purpose |
| --- | --- |
| `questions.jsonl` | Questions, structured ground truth, evidence references, and evaluator rules |
| `patient_splits.json` | Patient-to-split assignments |

### Build and validate

Run from the repository root. Dataset commands use the Python standard library;
Slurm launchers target Yale Bouchet. Data and artifacts are excluded from Git.
For a fresh clone, generate the source cohort first:

```bash
python3 -m dataset.cohort download
sbatch scripts/generate_cohort.sbatch 1
```

Once generation finishes, or with preserved source bundles available:

```bash
python3 -m dataset.questions --output data/processed/cohort-200 --replace
python3 -m dataset.validate --dataset data/processed/cohort-200
```

### Verify against live FHIR

```bash
python3 -m dataset.live prepare
sbatch scripts/verify_live_fhir.sbatch
```

Preparation requires Apptainer. The job uses Java 21 and PostgreSQL 16.4 to import
complete charts, check pagination, and recompute answers through HTTP. Failed
requests remain errors. Indexes, servers, and operational artifacts are temporary;
logs remain in `logs/`.

## Zero-shot pilot

The pilot selects **56 training questions from 44 patients**: four per family,
including eight missing-record cases. Validation and test data are reserved;
this pilot checks agent behavior rather than estimating held-out performance.

Qwen3-8B uses native tool calling with thinking disabled. Inputs contain the question,
patient reference, shared instructions, and a question-specific public answer schema. Tools provide patient-scoped
search, reference reads, pagination, and answer submission. Ground truth stays outside
the model.

Interface version 2 adds explicit FHIR query syntax and corrective parameter
errors. Submitted answers must match a public task structure before an episode
can finish; every episode uses the same validator without evaluator labels.
This checks formatting, not clinical correctness or search completeness. New run
headers record the interface version and validator checksum; preserve older
results and use a distinct `--output` path when comparing versions.

Interface version 3 also accepts a valid bare JSON tool call with `name` and
`arguments`, removes nearest-task guesses from schema feedback, and clarifies
report effective times and requested rounding. An identical failed action or
empty search triggers a warning on its third occurrence and ends the episode
with `repeated_failure` on its fourth occurrence without new retrieved resources.
Retrieving a new resource resets these counters. Valid nonempty repeated reads
do not trigger this guard. These are interface changes; answer scoring is unchanged.

Interface version 4 selects one public answer structure using only the visible
question wording, places its JSON Schema in the user message and `finish` tool,
and validates submissions against that structure. Unknown question wording fails
before GPU loading rather than falling back to any of the fourteen task shapes.
No family labels, ground truth, evidence IDs, or oracle metadata select the schema.
Headers store each selected contract and the policy for reconstructing its tool definition.

Search validation now rejects wrong-resource status values, contradictory repeated
status constraints, sort fields outside the supported resource-specific subset,
and encounter references not yet observed in tool results. Invalid queries produce
errors, not empty successful searches. Legitimate queries with no matches still
return empty results; these alone do not justify chart-wide absence claims.
The offline fixture transport now applies supported code, date, status, class,
reference and sort filters, and rejects unsupported fixture behavior instead of
silently ignoring it. This transport is not a replacement for live HAPI testing.

V4 passed 18 live query smoke checks and its 14-question training diagnostic
completed with 3 correct answers and 2 correct with required evidence. These are
development results, not held-out evaluation.

Interface version 5 adds a decimal `calculate` tool for later-minus-earlier
differences with explicit half-even rounding. It takes only agent-supplied numeric
strings, makes no HTTP requests, and counts against the same total tool budget.
Tool-call totals now include computation as well as retrieval; report this change
when comparing with older interfaces. It does not automatically alter final answers.
Contradictory AND filters on single-valued encounter/class fields now return
actionable errors rather than successful empty searches; OR remains explicit.
Shared instructions emphasize full-history temporal anchors, qualifying encounters,
targeted retrieval, explicit indication links, rounding, and anchor citations.
No ground-truth labels or resource IDs are exposed, and scoring is unchanged.

V5 is covered by offline regression tests only; its effect on model accuracy is
unmeasured. Use a fresh output path for evaluation and provide the same interface
to comparison systems. V4 artifacts remain historical evidence of v4 behavior.

Interface version 6 makes tool results smaller so a single search no longer fills the
16K-token student context, which caused most teacher `context_limit` failures:

- `search_fhir` returns at most 20 resources per page (was 50); follow `next_page` for more.
- Results omit `subject` (patient scope is automatic) and `DiagnosticReport.issued`.
- Condition `clinicalStatus` and `verificationStatus` are shortened to their bare code.
- `code.text` is dropped when it repeats a coding display. DiagnosticReport `result`
  references lose their long `display` strings.
- The answer schema appears once, in the user message. `finish` is the same for every
  question.
- The `search_fhir` description drops guidance repeated elsewhere, and its parameter
  schema no longer lists every key.

Every resource id and reference is kept, so citations and scoring are unchanged. An
offline test re-derives all 14 families' ground-truth answers from v6 projections; it
restores only `subject` and the full status concepts. On the 28-question teacher
selection, the first turn shrinks from 4,298 to 2,976 student tokens on average, and
projected resources shrink by 19–40% depending on type. V6 changes what the model sees,
so trajectories and zero-shot results from v5 and v6 are not directly comparable.

```bash
python3 -m pilot.run plan        # Preview selection without a model download
bash scripts/prepare_pilot.sh   # Install dependencies and cache the pinned model
```

After live verification passes, submit the GPU run:

```bash
sbatch scripts/run_pilot.sbatch
```

The launcher requests one L40S GPU, eight CPU cores, and 64 GB host memory. It
starts isolated FHIR servers and verifies selected charts before answering.
See [requirements-pilot.txt](requirements-pilot.txt) for pinned dependencies.

One experiment file, **`outputs/pilot.jsonl`**, contains run settings, episode
transcripts, answers, scores, and a summary. Logs go to `logs/pilot-<job-id>.log`.
After the job finishes:

```bash
python3 -m pilot.run summarize
```

Add `--replace` to intentionally replace an existing pilot result.

## Scoring

- **Answers:** compare structured outputs deterministically, matching coded concepts,
  equivalent timestamps, supported unit aliases, and unordered lists with multiplicity.
  Numeric tolerance is `0.000001` in requested units. Missing records and missing
  values remain distinct.
- **Evidence:** report required resource retrieval and citation coverage separately
  from answer accuracy. Coverage alone does not prove a complete search or justified
  absence claim; review transcripts before selecting SFT trajectories.
- **Effort:** record tool calls, HTTP requests, input/output tokens, elapsed time,
  and errors. Context, step, and token limits produce explicit failure statuses.

## Repository layout

```text
configs/       Synthea, cohort, and HAPI settings
dataset/       Generation, answer rules, validation, and live verification
pilot/         Agent loop, FHIR tools, scoring, and runner
scripts/       Preparation and Slurm launchers
tests/         Dataset, FHIR, and agent regression tests
data/          Preserved raw cohort and processed dataset
vendor/        Pinned artifacts and cached model
outputs/       Current pilot result
logs/          Slurm logs
```

Run offline checks:

```bash
python3 -m unittest discover -s tests -q
```

Next steps: confirm live retrieval, inspect real pilot trajectories, then collect
correct, efficient trajectories for SFT and expand the cohort for evaluation.
