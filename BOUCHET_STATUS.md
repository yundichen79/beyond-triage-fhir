# Bouchet execution status

Updated October 21, 2026, at 14:19:53 UTC (10:19:53 America/New_York).
This is a technical handoff, not student-authored report prose or a submission.
Measured results, implementation changes, and proposed next actions are separated below.

## Repository and authorization

- Last Git observation: branch `main`, commit `5451b09` (`Remove abundant docs`, October 15).
  This is historical context, not a fresh Git check. Current branch/commit and full dirty
  state have not been rechecked because every Git command requires user approval.
- Known local implementation changes: `README.md`, `pilot/agent.py`, `pilot/tools.py`,
  `pilot/run.py`, `tests/test_pilot.py`, and new `pilot/contract.py`. This status file is new.
  `AGENTS.md` and `BOUCHET_HANDOFF.md` were previously observed as untracked.
- No commits, fetches, pulls, pushes, or remote GitHub updates were performed during
  preparation, experimentation, diagnosis, or these fixes.
- Only authorized repository data and the user's relevant Slurm jobs were inspected.
  No unrelated cluster directories or private datasets were inspected.
- No new jobs were submitted in this final inspection. The jobs listed below have
  previously established outcomes; the entire user's queue was not rechecked now.

## Dataset and live verification: measured

- Regenerated Synthea v4.0.0 batch 1 using the repository's pinned seeds and dates.
  `data/raw/batch-001/generation.json` records `status=complete`, `patient_count=200`,
  and `exit_code=0`; there are 202 source bundles including provider exports.
- Question construction produced 1,202 questions across 14 families, with all 200
  patients represented. Split counts match the prior documentation:

| Split | Patients | Questions |
| --- | ---: | ---: |
| Train | 151 | 899 |
| Validation | 22 | 145 |
| Test | 27 | 158 |

- Provisional complexity counts: 499 simple, 548 intermediate, 155 complex.
  Question construction reported 90 primary negative cases across families.
- `python3 -m dataset.validate --dataset data/processed/cohort-200` completed with
  **1,202 questions checked, zero errors**. This command's result was observed in
  the session; a separate persistent log for that invocation was not created.
- Source normalization covered 528,053 resources. The live verification import
  scope was 527,460 resources across 200 patient charts and referenced dependencies.
  These are different processing scopes, not interchangeable counts.
- Full live FHIR verification passed: **200/200 patients, 1,202/1,202 questions**.
  `outputs/live-report1-attempt1/status.json` records `passed=true` and
  `postgres_stopped=true`; `verification.json` records `valid=true` and no errors.
  The verification finished at `2026-10-20T19:04:28.526865+00:00`.
- Raw-source replay and clinical review were reported by the pre-existing README.
  They were not separately rerun in this session. Live verification and local
  deterministic validation should not be described as independent clinical review.

Dataset files:

- `data/processed/cohort-200/questions.jsonl`
- `data/processed/cohort-200/patient_splits.json`
- Questions SHA-256: `0aad4dfc849db5c9e45084da62223d4c58a5d4c302f42aa9e84dbdbb9afffd6b`

## Preparation and job outcomes

HAPI and model preparation are complete. The initial HAPI extraction was slow but
finished successfully; the proposed `/tmp` preparation change was never implemented.
The initial interactive dependency install failed with `Cannot allocate memory`;
preparation then succeeded in a CPU Slurm allocation, and `pip check` found no broken
requirements.

| Job | Purpose | Established outcome | Runtime / evidence |
| --- | --- | --- | --- |
| `30046622` | Generate 200 patients | Generation manifest complete, exit 0; terminal Slurm state not re-queried | `data/raw/batch-001/generation.json`, `logs/synthea-30046622.log` |
| `30069832` | Full live verification | `COMPLETED`, `0:0` | 36m 6s; `logs/fhir-30069832.log`, `outputs/live-report1-attempt1/` |
| `30070133` | Dependencies and pinned model | `COMPLETED`, `0:0` | 5m 49s; `logs/prepare-pilot-30070133.log` |
| `30070323` | Original 56-question pilot | `COMPLETED`, `0:0` | 1h 6m 47s; `logs/pilot-30070323.log` |
| `30087543` | Interface v2 diagnostic | `COMPLETED`, `0:0` | 20m 8s; `logs/pilot-30087543.log` |
| `30136434` | First v3 attempt | `FAILED`, `1:0`; uncorrectable CUDA ECC error | 30s; node `a1125u05n01`; `logs/pilot-30136434.log` |
| `30136604` | Second v3 attempt | `FAILED`, `1:0`; same ECC error | 33s; node `a1125u14n01`; `logs/pilot-30136604.log` |
| `30137442` | v3 retry excluding both failed nodes | `COMPLETED`, `0:0` | 12m 54s; node `a1125u07n01`; `logs/pilot-30137442.log` |

The two ECC failures occurred before episode output was created. They are hardware
failures, not scored model episodes. The successful retry retained its own log.

## Pilot settings and completeness

All three successful runs used Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, thinking disabled, bfloat16 on an
NVIDIA L40S, Torch `2.9.1+cu126`, and Transformers `4.57.6`.

Shared settings: seed 4880, temperature 0.7, top-p 0.8, top-k 20, 24 model steps,
48 tool calls, 16,384 context tokens, 2,048 new tokens per turn, and 16,384 total
output tokens per episode. V3 additionally stops an identical failed action or
empty search at its fourth occurrence without newly retrieved resources.

| Artifact | Selection | Verified completeness | Correct / required-evidence correct |
| --- | --- | --- | --- |
| `outputs/pilot.jsonl` | 56 training questions, 44 patients; four per family, eight negative cases | 56 unique episodes, one summary, `complete=true` | 0/56 / 0/56 |
| `outputs/pilot-interface-v2-diagnostic.jsonl` | 14 training questions, 13 patients; one per family | 14 unique episodes, one summary, `complete=true` | 0/14 / 0/14 |
| `outputs/pilot-interface-v3-diagnostic.jsonl` | Same 14 questions as v2 | 14 unique episodes, one summary, `complete=true` | 2/14 / 2/14 (14.3%) |

Each diagnostic question is also present in the original 56-question run. The
comparisons below use those same 14 question IDs, not all 56 original episodes.
Selected charts passed live verification before each pilot: 56 questions / 44
patients originally, and 14 questions / 13 patients in each diagnostic.

## Matched 14-question results

| Metric | Original interface | Interface v2 | Interface v3 |
| --- | ---: | ---: | ---: |
| Correct answers | 0 | 0 | 2 |
| Correct with required retrieval and citation coverage | 0 | 0 | 2 |
| Episodes retrieving any resources | 1 | 8 | 8 |
| Episodes retrieving all required resources | 0 | 2 | 3 |
| Submitted answers (`completed`) | 6 | 4 | 3 |
| Step-limit terminations | 8 | 10 | 0 |
| Repeated-failure terminations | 0 | 0 | 11 |
| Model calls | 238 | 266 | 146 |
| Tool calls | 184 | 94 | 97 |
| Tool errors | 98 | 11 | 21 |
| Protocol errors | 48 | 168 | 46 |
| Output tokens, total | 31,148 | 28,251 | 16,489 |

For the full original 56-question run: 23 submitted answers, 33 step-limit failures,
516 tool errors, 175 protocol errors, mean 13.75 tool calls per episode, 3,323,119
input tokens, and 110,689 output tokens. These full-run counts must not be mixed
with the matched subset in the table.

Protocol counters include different checks across interfaces: v2/v3 add schema
validation. V2 had 64 malformed-call errors and 100 answer-schema rejections;
v3 had one malformed-call error and 44 answer-schema rejections. Their remaining
protocol errors were citation/envelope errors. V3 emitted 16 repeat warnings.

These runs are training-set development diagnostics, not held-out performance or
a fixed-workflow versus SFT comparison. Multiple interface changes were bundled;
the comparison does not isolate the causal effect of any individual change.
`complete=true` means every selected episode was recorded, not that every episode
submitted an answer. Evidence coverage does not prove exhaustive search or a
justified absence claim. No SFT trajectories have been approved or selected.

## Implementation changes and offline checks

- V2: explicit FHIR query guidance, typed parameter values, actionable parameter
  errors, and public answer-structure validation. Validation receives no ground
  truth, family labels, or oracle specifications. It accepts any public task shape;
  it does not establish that the chosen shape answers the particular question.
- V3: strict bare JSON tool-call support; removal of misleading nearest-task
  suggestions; explicit effective-time and half-even rounding instructions; bounded
  identical-failure handling with an explicit `repeated_failure` status.
- Clinical scoring was not relaxed. Its file checksum was checked against the
  original run header. Current implementation fingerprints were checked against
  the v3 header and matched the recorded files.
- Offline suite: 74 tests passed initially, 79 after v2, **83 after v3**.
  Latest log: `logs/interface-v3-tests.log`. Simulated failure messages inside that
  log are test fixtures; the final unittest result is `OK`.
- Offline transcript replay: 63 previously rejected calls now parse as valid bare
  tool-call JSON; one truly malformed call remains rejected. All 100 recorded
  invalid answer structures remain rejected without nearest-task suggestions.
  Evidence: `logs/interface-v3-replay.json`. This is parser/validator replay, not
  new model performance. Both earlier pilot files remained unchanged during it.
- V3 output SHA-256:
  `b0cac8d86cbce9bf199e83d05ed0ff1164f9c481988eb1739921028ab1d4a0e3`.

## Remaining failures: transcript observations

The 11 v3 `repeated_failure` terminations were classified from the terminal tool
response: nine repeated invalid answer structures, one repeated empty search,
and one repeated invalid date filter. The twelfth incorrect episode submitted a
format-valid but unsupported answer. These are observed behaviors, not hypotheses
about the model's internal reasoning.

- **CBC successes:** `cbc_change_3a381d12b9de14c7cdb2` and
  `cbc_panel_1ae9e837908a24caa305` were correct with required evidence coverage in v3.
  In v2 the comparison failed requested rounding, and the panel used `issued`
  instead of the report's effective time. Scoring stayed unchanged.
- **Wrong temporal restrictions:** discharge follow-up repeatedly searched inpatient
  encounters only after an invented 2024 lower boundary. Renal follow-up used
  invented recent-history bounds and `Encounter/AMB` as if it were a real reference.
- **Wrong resource statuses:** blood-pressure retrieval applied observation statuses
  (`final`, `amended`, `corrected`) to Encounter searches; procedure retrieval used
  those statuses for Procedure. Empty results were followed by absence claims.
- **Invented references and values:** the submitted inpatient-conditions answer used
  unsupported dates; searches included `Encounter/123456789` and invalid sort fields.
  No required evidence was retrieved for that answer.
- **Evidence retrieval alone is insufficient:** glycemic follow-up retrieved all
  required resources but proposed a 2021 result instead of the required first-year
  2008 follow-up after 2007 onset. Its answer also nested onset, baseline, and change
  under incorrect fields, then repeated the rejected structure.
- **Invalid absence and query syntax:** preadmission history invented a condition
  code and repeatedly used the string `null` as an abatement date. The date validator
  rejected it before HTTP. Medication questions still misplaced `count`, used
  `date` instead of `authoredon`, or submitted only `orders`/`products` without anchors.

Primary remaining blockers are task-specific temporal reasoning, valid clinical
search construction, evidence-grounded answers, and adherence to the public schema.
The v3 guard bounds repeated failures; it does not repair them or make an absence
claim valid. The GPU ECC errors were bypassed on another node, not repaired.

## Reproduction references (commands are documentation, not new authorization)

```bash
python3 -m unittest discover -s tests -q
python3 -m pilot.run plan
python3 -m pilot.run summarize --output outputs/pilot.jsonl
python3 -m pilot.run summarize --output outputs/pilot-interface-v2-diagnostic.jsonl
python3 -m pilot.run summarize --output outputs/pilot-interface-v3-diagnostic.jsonl
```

Historical successful v3 submission:

```bash
sbatch --exclude=a1125u05n01,a1125u14n01 scripts/run_pilot.sbatch \
  --per-family 1 --output outputs/pilot-interface-v3-diagnostic.jsonl
```

Do not rerun into an existing output, replace evidence, submit another job, or use
Git without the user's approval. A future run needs a fresh output path. Use run
headers for model, settings, dataset and source identities, prompt, tool definitions,
and code hashes; use episode IDs for matched comparisons. Summaries aggregate
available episodes, so also check unique counts and the final `complete` field.

## Report preparation and recommended next action

Recommended next action: preserve this checkpoint and use the measured figures to
correct the student's existing draft before another full GPU run. A further
implementation change or experiment needs separate approval. If more development
is authorized, first address selecting the requested public structure from the
question, correct per-resource search semantics, and explicit temporal anchors;
do not infer that another full 56-question run will fix these blockers.

Factual corrections/checks for Jun (not a drafted report paragraph):

- Replace claims that live verification/model preparation/real GPU execution are
  still pending. README's original current-status table is stale; it was not
  rewritten during this final status-only task.
- Distinguish the full original pilot's 0/56 from v3's 2/14 on a training diagnostic
  subset. Do not report 14.3% as a 56-question or held-out result.
- Describe retrieval improvements and continuing failures; avoid claiming the
  adaptive agent is already accurate, efficient relative to a fixed baseline, or
  ready for SFT solely because two trajectories passed evidence scoring.
- Fixed-workflow comparison, SFT training, dataset expansion, and held-out
  comparative results remain unavailable from the inspected artifacts.
- Keep numeric totals versus means explicit and missing-record cases within their
  existing episode denominator. Separate provisional difficulty from measured difficulty.
- The actual draft, staff template, Canvas deadline, contribution statement, AI
  disclosure, page count, and exported submission file were not inspected in this
  task. They still require Jun's review or explicit authorization to inspect the
  relevant materials. No coursework was submitted and no teammates were contacted.
- The handoff states students must write substantive report prose and disclose AI
  assistance. This file supplies technical facts and references for that review;
  it does not complete submission preparation or independently verify course policy.
