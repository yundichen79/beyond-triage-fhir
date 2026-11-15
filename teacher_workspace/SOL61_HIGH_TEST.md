# GPT-6.1 Sol High comparison

The user requested about 28 training questions across all 14 families before
deciding which teacher to use for the remaining corpus. This is a bounded test,
not approval to switch the entire corpus.

Job **33486023** was submitted on November 15, 2026. Its initial observed state
was RUNNING, preparing the synthetic source charts. Check current artifacts for
results; submission alone does not establish completion.

## Frozen selection

`teacher_workspace/selections/sol61-high-28-v1.json` contains 28 training episode
IDs: two per family, 25 patients, eight primary negative cases. Eighteen questions
have prior non-runtime-error attempts in `outputs/teacher-batches/batch-002`.

Within each family, reserve one negative when available and fill the remaining
slot(s) with positives. Within these strata, prefer previous attempts, then sort
by opaque episode ID. Correctness was not used for selection. Questions run in two
rounds over all families. This manifest is operator-only and is never provided to
an individual teacher. Dataset identity is checked before execution.

## Model and launch

The Codex command explicitly supplies `--model gpt-6.1-sol` and
`-c model_reasoning_effort="high"`. Both requested values are recorded in the batch
manifest, per-question provenance, and trajectories. No implicit model selection
is used for this test. These record the requested settings, not a server-returned
model fingerprint.

```sh
sbatch --parsable scripts/run_teacher_batch.sbatch \
  --selection teacher_workspace/selections/sol61-high-28-v1.json \
  --teacher-model gpt-6.1-sol --reasoning-effort high \
  --sessions teacher_workspace/batches/sol61-high-28-v1 \
  --output outputs/teacher-batches/sol61-high-28-v1
```

The same public instructions, patient tools, scoring, and student transcript
budgets apply. Each question starts a fresh ephemeral Codex process. Existing
ChatGPT login is required; API-key overrides are removed. No credentials are
copied into teacher sessions or displayed. Slurm allocation: account `cpsc4880`,
standard `day`, QoS `normal`, four CPUs, 24 GB RAM, four hours, no GPU.

Pre-submission checks: 108 offline tests passed, launcher syntax passed, and
Slurm test-only accepted the allocation. Logs: `logs/teacher-sol61-high-tests.log`.

## Results

Job log: `logs/teacher-batch-33486023.log`.
Private results: `outputs/teacher-batches/sol61-high-28-v1/`.

To build or refresh the comparison after recorded attempts:

```sh
python3 -m teacher.report --output outputs/teacher-batches/sol61-high-28-v1
```

This writes `COMPARISON.md` and `comparison.json`, separating all selected questions
from matched comparisons and explicitly marking incomplete results and missing
Codex usage events. Do not infer allowance percentage from student token counters.

The prior teacher was not pinned; the user's identification as Astra High is not
independently established by its logs. Preserve that limitation when comparing.
Do not infer held-out performance from this small training diagnostic, or treat
evidence coverage alone as approval for SFT.
