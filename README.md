# Beyond TRIAGE-FHIR

Adaptive question answering over FHIR electronic health records, extending
[TRIAGE-FHIR](https://github.com/yundichen79/triage-fhir).

**Question:** does supervised fine-tuning (SFT) on teacher trajectories make a small
tool-using model better or more efficient at chart-review questions?

**Systems compared,** all in the same FHIR environment with the same tools and budgets:
1. the fixed TRIAGE-FHIR workflow
2. a zero-shot Qwen3-8B agent
3. Qwen3-8B fine-tuned on verified teacher trajectories

**Metrics:** answer accuracy, accuracy with required evidence, tool calls and tokens.

See the [question catalog](data/README.md) for dataset rules, fields and examples, and
the [teacher guide](teacher/README.md) for trajectory generation.

## Pipeline and status

| Stage | Code | Status |
| --- | --- | --- |
| 1. Synthetic cohort and questions | `dataset/` | Done: 200 patients, 1,202 questions, validated and clinically reviewed |
| 2. Live FHIR verification | `dataset/live.py` | Done: 200 patients / 1,202 answers recomputed through HTTP |
| 3. Student environment | `student/` | Interface v6; used live in every teacher episode |
| 4. Teacher trajectories | `teacher/` | Done for now: Claude Sonnet 5.5 (high effort), 306 grounded trajectories for 318 of 899 training questions |
| 5. SFT data | `sft/data.py` | Done: clean (1,179 examples) and recovery (1,171) variants |
| 6. SFT training | `sft/train.py` | Ready: LoRA script, launcher and dry runs pass; GPU smoke test queued |
| 7. Baselines and evaluation | `student/run.py` | Runner handles validation/test and LoRA adapters; v6 zero-shot on validation and test queued. Earlier zero-shot: training diagnostics only (v4/v5: 3 of 14 correct) |

The [next steps](#next-steps) section lists what remains.

## Dataset

**200 synthetic Massachusetts patients** were generated with Synthea v4.0.0 as FHIR R4
bundles with complete histories:
- seeds and dates: [cohort.json](configs/cohort.json)
- exporter settings: [synthea.properties](configs/synthea.properties)
- original bundles and their generation record with checksums: `data/raw/batch-001/`

The dataset has **1,202 questions across 14 clinical chart-review families**, with an
exclusive record cutoff of **2026-10-02 UTC**. Tasks cover encounter-linked conditions
and orders, CBC and blood pressure, documented indications, medication comparisons and
longitudinal follow-up. Every answer is computed by deterministic rules, with explicit
time windows, units and supporting resource references. Difficulty labels are
provisional: 499 simple, 548 intermediate, 155 complex.

| Split | Patients | Questions |
| --- | ---: | ---: |
| Train | 151 | 899 |
| Validation | 22 | 145 |
| Test | 27 | 158 |

All questions for a patient stay in the same split. Only two processed files are kept,
in `data/processed/cohort-200/`:
- `questions.jsonl`: questions, structured ground truth, evidence references and
  evaluator rules
- `patient_splits.json`: patient-to-split assignments

**Construction:**
1. Preserve the Synthea bundles and normalize references in a temporary SQLite index.
2. Generate eligible questions and answers, including sampled missing-record cases.
3. Validate answers, evidence, question rules and patient splits.
4. Replay answers from the raw bundles, and review them clinically.

Build and check, from the repository root (dataset commands need only the Python
standard library; launchers target Yale Bouchet):

```bash
python3 -m dataset.cohort download             # fresh clone only
sbatch scripts/generate_cohort.sbatch 1        # fresh clone only
python3 -m dataset.questions --output data/processed/cohort-200 --replace
python3 -m dataset.validate --dataset data/processed/cohort-200
python3 -m dataset.live prepare                # needs Apptainer
sbatch scripts/verify_live_fhir.sbatch         # import charts, recompute every answer over HTTP
```

**Known limitation:** `followup_medication_comparison` has two wording and rule problems.
- Its question doesn't ask for elapsed days rounded to one decimal, but its label is
  rounded.
- When there is no follow-up, its evidence rule still requires every inpatient order
  and Medication record.

Both are left unchanged so the evaluated benchmark stays fixed. As a result the family
contributes only one teacher trajectory.

## Student environment

`student/` is the environment every system answers in. A model gets:
- the question and patient reference
- shared instructions
- one public answer schema chosen from the question wording

It can use:
- patient-scoped `search_fhir`
- `read_fhir` for references already seen
- `next_page`
- a decimal `calculate`
- `finish`, which must cite retrieved resources

Ground truth never reaches the model. Episodes are limited to 24 turns, 48 tool calls
and a 16,384-token transcript. Qwen3-8B uses native tool calling with thinking disabled.

Interface history (each version changes what the model sees, so results from different
versions are not directly comparable):

- **v2:** explicit FHIR query syntax, corrective parameter errors, and answers
  validated against a public structure.
- **v3:** bare JSON tool calls accepted. Repeated identical failures warn on the third
  occurrence and end the episode on the fourth.
- **v4:** answer schema chosen from the question wording alone. Searches with
  wrong-resource statuses, contradictory filters, unsupported sorts or unseen
  encounters are rejected. 18 live query checks passed.
- **v5:** a decimal `calculate` tool and revised retrieval instructions.
- **v6 (current):** smaller tool results, so one search can no longer fill the student
  context.
  - Pages hold at most 20 resources.
  - Results drop `subject`, `DiagnosticReport.issued` and long result `display` strings.
  - Status concepts are reduced to bare codes, and `code.text` is dropped when it
    repeats a display.
  - The answer schema appears once, in the user message.

  Every id and reference is kept, so scoring is unchanged. An offline test re-derives
  all 14 families' answers from v6 results. The first turn shrank from 4,298 to 2,976
  tokens on average.

Running Qwen3-8B:

```bash
bash scripts/prepare_model.sh                     # install torch/transformers/peft, cache the pinned model
python3 -m student.run plan --split validation    # preview a selection
sbatch scripts/run_student.sbatch --split test --output outputs/zero-shot-v6-test.jsonl
python3 -m student.run summarize --output outputs/zero-shot-v6-test.jsonl
```

- **`--split`:** `train` is the 56-question diagnostic (4 per family); `validation`
  and `test` run every question in that split. One run covers one split.
- **`--adapter <run>/epoch-N`:** evaluates a fine-tuned model. The run header then
  records `"mode": "sft"`.
- **Hardware:** one A40 or L40S (48 GB), 8 CPUs and 64 GB memory on the free `gpu`
  tier.
- **Output:** results go to the `--output` file (default `outputs/zero-shot.jsonl`),
  and logs to `logs/student-<job-id>.log`.
`sbatch scripts/verify_student_queries.sbatch` reruns the live query checks.

## Teacher trajectories

The teacher solves training questions through the same student environment, so every
trajectory is already in the student's format.
- **Process:** each question runs in a fresh Claude Code process on a subscription
  login (API-key billing is refused). The teacher's system prompt is
  `teacher/TEACHER.md`, and it may run only `teacher_client.py`.
- **Scoring:** answers are scored privately after each episode.
- **Running out of usage:** the batch waits out 5-hour usage limits and resumes where
  it stopped.

Details are in the [teacher guide](teacher/README.md).

On the 28-question v6 selection (two per family), Claude Sonnet 5.5 at high effort
answered 25 correctly with full evidence. The current corpus is
`outputs/teacher-corpus/sonnet55-v6-20261116/`, built with `teacher.collect`:

| | |
| --- | --- |
| Training questions with a finished attempt | 318 of 899 (questions run in random order) |
| Grounded trajectories (`selected`) | 306: 142 simple, 126 intermediate, 38 complex; 28 missing-record cases |
| Families | All 14; thinnest: `followup_medication_comparison` 1, `admission_medication_comparison` 3, `glycemic_followup` 4, `renal_followup` 6, `cbc_change` 9 |
| Mean tool calls per selected trajectory | 3.9 |
| Teacher input tokens per question | about 81K, mostly cached |

## SFT data

`sft.data` turns the selected trajectories into training examples:

```bash
python3 -m sft.data --corpus outputs/teacher-corpus/sonnet55-v6-20261116 \
  --output outputs/sft-data/sonnet55-v6-20261116 [--mode recovery]
```

- **Format:** one example per assistant turn.
  - `prompt` is the exact text the student sees before that turn: the pinned Qwen3-8B
    chat template, generation prompt on, thinking off.
  - `completion` is the turn in Qwen's native `<tool_call>` format, ending with
    `<|im_end|>`.
  - Train with loss on the completion only.
- **Failed actions:** 110 trajectories contain a failed action the teacher recovered
  from (57 malformed turns, 89 failed tool calls). No trajectory is dropped, and no
  failed action is ever a training target.
  - `clean` mode removes failed actions and their error feedback.
  - `recovery` mode keeps the conversation as recorded, so the corrected turns that
    follow are trained in the context of the failure.
- **Checks:**
  - Prompts fit the 14,336-token budget; the longest is 13,784.
  - Prompt token counts match those recorded at inference (all 1,171 recovery prompts).
  - Every completion parses with the student's tool-call parser.
- **Dev split:** about 10% of patients form a dev set for training loss. The dataset's
  validation split stays reserved for choosing settings.

| Variant | Folder under `outputs/sft-data/` | Train / dev examples | Recovery examples |
| --- | --- | ---: | ---: |
| Clean | `sonnet55-v6-20261116` | 1,098 / 81 | 0 |
| Recovery | `sonnet55-v6-20261116-recovery` | 1,091 / 80 | 221 |

Each training set is about 5.2M prompt tokens and 173K completion tokens.

## SFT training

`sft/train.py` fine-tunes Qwen3-8B with LoRA on one data variant:
- **Loss:** completion tokens only. Predictions are computed only for those positions,
  which keeps memory low on prompts of up to about 14K tokens.
- **Setup:** bf16 weights, LoRA on every attention and MLP projection, gradient
  checkpointing.
- **Defaults:** 2 epochs, learning rate 1e-4 with warmup and cosine decay, 16 examples
  per optimizer step (138 steps in total), LoRA rank 16 / alpha 32.

Each run writes, to a new folder under `outputs/sft-runs/`:
- `train-config.json`: settings, data fingerprints and library versions
- `train-log.jsonl`: training and dev losses
- `epoch-<n>/`: one adapter per epoch

`--dry-run` checks the data and tokenization without a GPU. It passed on both
variants: every prompt and completion tokenizes the same apart as joined, and the
longest example is 14,269 tokens. `--smoke` runs 2 steps on the longest examples.
`scripts/train_sft.sbatch` launches training on one A40 or L40S (up to 24 hours, free
tier).

## Scoring

- **Answers:** structured outputs are compared deterministically. Coded concepts match
  by system and code, and timestamps, unit aliases and unordered lists are compared
  with multiplicity. Numeric tolerance is `0.000001` in the requested units. Missing
  records and missing values stay distinct.
- **Evidence:** retrieval and citation of the required resources are reported
  separately from accuracy. *Grounded* means correct with all required evidence
  retrieved and cited.
- **Effort:** tool calls, HTTP requests, input/output tokens, elapsed time and errors.
  Context, step and token limits give explicit failure statuses.

## Next steps

1. **Check the SFT setup on a GPU.** Run the smoke test (queued): 2 optimizer steps on
   the 16 longest examples. It confirms peak GPU memory, speed and adapter saving.
   ```bash
   sbatch scripts/train_sft.sbatch --smoke \
     --data outputs/sft-data/sonnet55-v6-20261116-recovery --output outputs/sft-runs/smoke-recovery
   ```
2. **Spot-check the SFT data** while training runs. Read a few selected trajectories
   per family for invented filters, wrong anchors and unsupported missing-record
   claims. Grounding is checked automatically; search quality is not.
3. **Train both variants** with the default settings: 2 epochs, learning rate 1e-4,
   LoRA rank 16, 16 examples per step. That's one model on the clean data and one on
   the recovery data, giving the clean-versus-recovery ablation:
   ```bash
   sbatch scripts/train_sft.sbatch --data outputs/sft-data/sonnet55-v6-20261116-recovery \
     --output outputs/sft-runs/recovery-r16-lr1e-4
   sbatch scripts/train_sft.sbatch --data outputs/sft-data/sonnet55-v6-20261116 \
     --output outputs/sft-runs/clean-r16-lr1e-4
   ```
4. **Choose settings on the validation split.** Run each epoch's adapter on all 145
   validation questions and pick the variant and epoch by grounded accuracy. Try
   another learning rate or epoch count only if validation suggests it.
   ```bash
   sbatch scripts/run_student.sbatch --split validation \
     --adapter outputs/sft-runs/recovery-r16-lr1e-4/epoch-2 --output outputs/sft-recovery-e2-validation.jsonl
   ```
5. **Zero-shot baseline on v6** (queued). Qwen3-8B without an adapter, on the full
   validation and test splits, writing to `outputs/zero-shot-v6-{validation,test}.jsonl`.
6. **Fixed-workflow baseline.** Port the TRIAGE-FHIR workflow onto the same environment,
   tools and budgets.
7. **Final test evaluation.** Freeze the chosen SFT settings, then run all three
   systems on the 158 test questions. Compare:
   - accuracy, grounded accuracy and per-family (macro) accuracy
   - tool calls and tokens, and effort by difficulty
   - with patient-level bootstrap confidence intervals

   Each system is one sampled run (temperature 0.7, seed 4880); say so in the report.
8. **Report.** State the limitations:
   - synthetic data
   - a partial corpus (318 of 899 training questions)
   - the `followup_medication_comparison` issues
   - teacher model identity taken from the requested model name, not a returned
     fingerprint
   - disclose the AI-generated trajectories

   Optional: top up the thin families (about 50 questions, under one usage window).

## Repository layout

```text
configs/       Synthea, cohort and HAPI settings; frozen question selections
dataset/       Cohort generation, questions and answer rules, validation, live verification
student/       Student environment: agent loop, FHIR tools, answer contracts, scoring, runner
teacher/       Teacher trajectory generation, comparison reports, corpus collection
sft/           SFT data conversion and LoRA training
scripts/       Preparation scripts and Slurm launchers
tests/         Offline regression tests (dataset, FHIR, student, teacher, SFT data)
data/          Raw cohort and processed dataset (data files not in Git)
vendor/        Pinned HAPI and Synthea artifacts and the cached model (not in Git)
outputs/       Runs, teacher batches, trajectory corpus and SFT data (not in Git)
logs/          Slurm logs (not in Git)
```

Run the offline tests from the repository root:

```bash
python3 -m unittest discover -s tests -q
```
