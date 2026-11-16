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
| Dataset | Built; local validation, source replay and clinical review passed |
| Live FHIR verification | Passed for 200 patients / 1,202 questions; v4 query checks passed 18/18 |
| Student interface | Version 6; used live in every teacher episode, live query check not yet rerun |
| Zero-shot Qwen3-8B | Training diagnostics on v4/v5: 3/14 correct; not yet run on v6 |
| Teacher trajectories | Claude Sonnet 5.5 (high effort): 306 grounded trajectories for 318 of 899 training questions; review pending |
| SFT data | 1,179 per-turn examples from the 306 grounded trajectories (1,098 train / 81 dev) |
| SFT training and evaluation | Planned: LoRA training, fixed-workflow comparison and held-out evaluation outstanding |

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

## Student environment and zero-shot runs

The `student/` package is the environment every model answers in: the zero-shot
Qwen3-8B baseline, the teacher that generates SFT trajectories, and the fine-tuned
student. Qwen3-8B uses native tool calling with thinking disabled. Its input is the
question, the patient reference, shared instructions, and a question-specific public
answer schema. Tools provide patient-scoped search, reference reads, pagination, a
decimal calculator and answer submission. Ground truth stays outside the model.

The zero-shot runner (`student.run`) selects **56 training questions from 44
patients**, four per family including eight missing-record cases. Validation and test
data are reserved, so these runs check agent behavior and do not estimate held-out
performance.

Interface history (each version changes what the model sees, so results from
different versions are not directly comparable):

- **v2:** explicit FHIR query syntax, corrective parameter errors, and answers
  validated against a public task structure before an episode can finish.
- **v3:** bare JSON tool calls accepted; repeated identical failures or empty searches
  warn on the third occurrence and end the episode (`repeated_failure`) on the fourth.
- **v4:** one public answer schema chosen from the question wording alone; search
  validation rejects wrong-resource statuses, contradictory filters, unsupported sort
  fields and unseen encounter references. 18 live query checks passed; the 14-question
  training diagnostic got 3 correct.
- **v5:** a decimal `calculate` tool (later minus earlier, half-even rounding) and
  revised retrieval instructions. Its 14-question training diagnostic got 3 grounded.

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
python3 -m student.run plan        # Preview the selection without a model download
bash scripts/prepare_model.sh      # Install dependencies and cache the pinned model
sbatch scripts/run_student.sbatch  # GPU run after live verification passes
python3 -m student.run summarize   # After the job finishes
```

The launcher requests one L40S GPU, eight CPU cores and 64 GB host memory. It starts
isolated FHIR servers and verifies the selected charts before answering. Pinned
dependencies are in [requirements-model.txt](requirements-model.txt). Results go to
`outputs/zero-shot.jsonl` (run settings, transcripts, answers, scores and a summary)
and logs to `logs/student-<job-id>.log`. Use `--output` for a new file, or `--replace`
to overwrite an existing one. `sbatch scripts/verify_student_queries.sbatch` reruns the
live query checks.

## Teacher trajectories

A teacher model solves training questions through the same student environment, so
every trajectory is already in the student's format. Each question runs in a fresh
Claude Code process (`claude -p`) on a subscription login; API-key billing is refused.
The process gets `teacher/TEACHER.md` as its system prompt and only one allowed
command, `teacher_client.py`. Answers are scored privately after each episode. See
[teacher/README.md](teacher/README.md) for launching, usage limits and resuming.

On the 28-question v6 selection (two per family), Claude Sonnet 5.5 at high effort
answered 25 correctly with full evidence. The current corpus is in
`outputs/teacher-corpus/sonnet55-v6-20261116/`:

| | |
| --- | --- |
| Training questions with a finished attempt | 318 of 899 |
| Grounded trajectories (`selected`) | 306, including 28 missing-record cases |
| Difficulty of selected (simple / intermediate / complex) | 142 / 126 / 38 |
| Mean tool calls per selected trajectory | 3.9 |

`teacher.collect` builds the corpus from batch runs with identical student-interface
code. It marks the shortest grounded attempt per question and leaves out hidden
labels. Selected trajectories still need review of anchors, filters and absence
claims before SFT.

**Known limitation:** `followup_medication_comparison` contributes only one trajectory.
Its question text does not ask for the elapsed days to be rounded to one decimal,
though its label is rounded. And when there is no follow-up, its evidence rule still
requires every inpatient order and Medication record. Both are left unchanged so the
evaluated benchmark stays fixed.

## SFT data

`sft.data` converts the selected trajectories into training examples:

```bash
python3 -m sft.data --corpus outputs/teacher-corpus/sonnet55-v6-20261116 \
  --output outputs/sft-data/sonnet55-v6-20261116
```

- **Cleaning:** failed actions retrieve nothing, so they are removed before
  conversion: malformed turns with their format-error feedback, and failed tool calls
  with their error results. Every remaining action and result is real. 110 of the
  306 trajectories needed cleaning (57 malformed turns, 89 failed calls).
- **Format:** one example per assistant turn. `prompt` is the exact text the student
  sees before that turn: the pinned Qwen3-8B chat template with
  `add_generation_prompt=True` and `enable_thinking=False`. `completion` is the turn in
  Qwen's native `<tool_call>` format, ending with `<|im_end|>`. Train with loss on the
  completion only.
- **Checks:**
  - Each prompt must be a prefix of the rendered conversation through that turn.
  - Prompts must fit the 14,336-token student budget; the longest is 13,784.
  - For all 196 trajectories that needed no cleaning, prompt token counts are
    identical to those recorded at inference.
  - Every completion parses with the student's tool-call parser.
- **Split:** about 10% of patients are a dev set for training loss: 1,098 train / 81
  dev examples from 120 / 9 patients. The dataset's validation split stays reserved
  for choosing settings by agent runs.

Output: `train.jsonl`, `dev.jsonl` and `sft-data.json` (provenance, cleaning counts and
token totals). The training set is 5.2M prompt tokens and 174K completion tokens.

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
configs/       Synthea, cohort and HAPI settings; frozen question selections
dataset/       Generation, answer rules, validation and live verification
student/       Student environment: agent loop, FHIR tools, answer contracts, scoring, runner
teacher/       Teacher trajectory generation, reports and corpus collection
sft/           Conversion of teacher trajectories into SFT examples
scripts/       Preparation and Slurm launchers
tests/         Dataset, FHIR, agent and teacher regression tests
data/          Preserved raw cohort and processed dataset
vendor/        Pinned artifacts and cached model
outputs/       Run results, teacher batches and the trajectory corpus (not in Git)
logs/          Slurm logs
```

Run offline checks:

```bash
python3 -m unittest discover -s tests -q
```

Next steps: LoRA training on the SFT data with settings chosen on the validation split,
the v6 zero-shot baseline, and a test-split comparison of fixed workflow, zero-shot and SFT.
