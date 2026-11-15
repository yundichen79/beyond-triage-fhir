# Teacher batch restart

Observed 2026-11-15 at 14:59 UTC. Operator-only context; do not supply this file
to an individual teacher session.

## Previous run

- Interactive pilot `4d772a3230da8ecc5ad9bc61-attempt1` has a saved correct answer
  with required evidence. `approved_for_sft` remains false pending review.
- Batch job `33438022` verified 898 questions across 151 synthetic patients, then
  hit the Codex account usage limit during its first question, after two tool calls.
  Its summary records one attempted question, zero grounded successes, and
  `teacher_runtime_error`. This is an interrupted attempt, not a completed corpus.
- Both previous jobs (`33436622`, `33438022`) were absent from the active queue
  before restart. All previous outputs were preserved.

## Fix and checks

`teacher.batch.select_rows` now retries saved `model_error` attempts in a new
session instead of treating their evaluation file as completed work. Other scored
attempts remain skipped regardless of correctness. A later completed retry takes
precedence regardless of resume-source order. Dataset identity checks remain intact.

- Full offline suite: **106 tests passed**; log:
  `logs/teacher-resume-tests-20261115.log`.
- Resume plan: 898 questions, 151 patients, fresh context per question.
- Launcher syntax and `git diff --check` passed.
- Existing ChatGPT login confirmed without exposing credentials. No API-key mode.
- Slurm test-only submission accepted the configured allocation.

## Restart

Submitted with the user's approval:

```sh
sbatch --parsable scripts/run_teacher_batch.sbatch \
  --sessions teacher_workspace/batches/batch-002 \
  --output outputs/teacher-batches/batch-002 \
  --resume-from outputs/teacher/4d772a3230da8ecc5ad9bc61-attempt1 \
  --resume-from outputs/teacher-batches/batch-001
```

Job **33467807** was observed **RUNNING** on `c1106u11n01`. Its log reached
`Loading and checking source records`; no new teacher result was available at
this checkpoint. This does not yet establish that inference can proceed without
another account limit. The batch stops on a teacher runtime error.

Allocation: account `cpsc4880`, partition `day`, QoS `normal`, four CPUs, 24 GB RAM,
four-hour limit, no GPU. YCRC's published standard-tier no-charge computation
policy was checked before submission:
https://research.computing.yale.edu/computing-resources/hpc-policies
Codex uses the existing signed-in account allowance. No credits were purchased.

Progress artifacts:

- `logs/teacher-batch-33467807.log`
- `outputs/teacher-batches/batch-002/summary.json` (written after attempts or shutdown)
- `outputs/teacher-batches/batch-002/results.jsonl`
- `outputs/teacher-batches/batch-002/attempts/`

The 3.5-hour soft budget includes backend setup; this allocation is not expected
to finish all 898 questions. Check the job and artifacts before submitting any
continuation. Use fresh paths and include all previous resume sources. Keep scores
private, retain failed trajectories, and review successes before SFT selection.
