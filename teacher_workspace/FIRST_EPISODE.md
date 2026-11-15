# First teacher episode

Training-only pilot episode: `4d772a3230da8ecc5ad9bc61`. This example is one CBC question, not a
representative dataset-wide evaluation. Use fresh contexts for later questions.

After approval and verification of CPU billing eligibility, run from the repository
root:

```bash
sbatch scripts/run_teacher.sbatch --episode 4d772a3230da8ecc5ad9bc61 --session teacher_workspace/sessions/4d772a3230da8ecc5ad9bc61-attempt1 --output outputs/teacher/4d772a3230da8ecc5ad9bc61-attempt1
```

The teacher's future working directory is:

`teacher_workspace/sessions/4d772a3230da8ecc5ad9bc61-attempt1`

Wait for `state.json` to say `awaiting_response`. Open a fresh conversation restricted
to that directory and ask it to follow AGENTS.md and solve episode.json through
teacher_client.py. Job 33436622 has now launched this attempt; do not resubmit it. See ACTIVE_SESSION.md.
