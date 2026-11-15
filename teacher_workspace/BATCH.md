# Generate trajectories for the training dataset

Latest restart: see [RESUME_STATUS.md](RESUME_STATUS.md) for job `33467807`,
the usage-limit interruption, and the runtime-error resume fix.

The one-question interactive session was a pilot. The batch runner covers **all
899 training questions**, using a fresh `codex exec` process for each question.
One conversation must not solve the whole dataset: later questions would inherit
earlier patient records and answers.

## What happens automatically

1. Select every remaining training question; validation and test are excluded.
2. Start and verify one live FHIR backend for the selected patient charts.
3. For each question, create a new public session with only its question, patient,
   shared instructions, answer schema, tools, budgets, and public client.
4. Start a fresh, ephemeral Codex process with filesystem access restricted to that
   session and the Codex runtime. No resume or fork is used; user config is not
   loaded. Managed host configuration still applies. The model is the executable's
   default, not a model override; inspect recorded CLI events to establish its identity.
5. Execute actual tool actions through the unchanged student agent loop, record the
   trajectory, and score it privately after termination. Then end that Codex process.
6. Save progress after every question. A later allocation can skip recorded attempts
   without overwriting earlier results, including the original interactive pilot.

The question-level loop still stops after one question. The **batch orchestrator**
starts the next fresh teacher; the teacher prompt should not ask one context to
continue through all patients.

## Prepared launch command

Run only after approval for the CPU allocation and Codex account usage:

```bash
sbatch scripts/run_teacher_batch.sbatch \
  --sessions teacher_workspace/batches/batch-001 \
  --output outputs/teacher-batches/batch-001 \
  --resume-from outputs/teacher/4d772a3230da8ecc5ad9bc61-attempt1
```

The current plan selects **898 remaining questions**: the first interactive attempt
already has a saved evaluation and is preserved. Skipping a recorded attempt does
not mean its answer was correct or approved for SFT.

This launch requests account `cpsc4880`, standard partition `day`, QoS `normal`,
4 CPUs, 24 GB RAM, and 4 hours. Standard-tier cluster computation is free under
YCRC's published policy; recheck eligibility if that policy/configuration changes.
No GPU or model-weight download is used. Existing ChatGPT sign-in is required;
the runner removes API-key environment overrides and refuses API-key login mode.
Codex requests consume the signed-in account's allowance and can encounter limits.
No authentication is copied into the teacher-visible directories.

## Time limits and continuation

The soft run budget is 3.5 hours including FHIR setup, with at most 15 minutes per
question. The runner stops between episodes before starting one that would exceed
that budget. **Do not expect 898 questions to finish in a single four-hour job.**
It also stops the batch on a teacher startup/auth/runtime error rather than
launching hundreds of failing processes. Inspect private logs before continuing.

For a subsequent approved allocation, use new paths and list every earlier batch
whose recorded attempts should be skipped:

```bash
sbatch scripts/run_teacher_batch.sbatch \
  --sessions teacher_workspace/batches/batch-002 \
  --output outputs/teacher-batches/batch-002 \
  --resume-from outputs/teacher/4d772a3230da8ecc5ad9bc61-attempt1 \
  --resume-from outputs/teacher-batches/batch-001
```

Incomplete attempts lacking evaluation and attempts ending in `model_error`
(including usage-limit interruptions) are retried in a new session. Other evaluated
attempts are skipped regardless of score. Deliberate retries of incorrect answers
should be a separate experiment with fresh paths, not overwritten evidence.
The runner verifies dataset identity before accepting resume sources.

Preview selection without inference or a Slurm submission:

```bash
python3 -m teacher.batch plan \
  --resume-from outputs/teacher/4d772a3230da8ecc5ad9bc61-attempt1
```

## Outputs and interpretation

Private batch output includes `batch.json`, `results.jsonl`, `summary.json`, live
verification logs, and `attempts/<opaque-id>/` containing the trajectory, raw
request/response journal, provenance, score, and Codex event/error logs.

`summary.json.complete` means every selected question has an evaluated attempt,
not that every answer succeeded. `approved_for_sft` defaults to false. Review
anchors, unsupported filters, references, and absence completeness before selecting
demonstrations. Token limits describe the Qwen-tokenized submitted transcript;
actual Codex usage, when available, is recorded separately in its event stream.

## Validation scope

Offline tests cover training-only selection, dataset-change rejection, resume,
fresh-process command construction, API-key exclusion, and a complete question
through the real agent loop with fixture transport. They do not invoke Codex
inference. Automatic Codex execution on a compute node must be verified during
the first approved batch; the earlier pilot used interactive Codex.

This document and operator logs must not be supplied to the individual teacher.
