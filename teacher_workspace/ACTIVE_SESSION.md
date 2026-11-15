# Active teacher session

Historical single-question session below. It has ended and its saved evaluation
is correct with required evidence, pending SFT review. The current batch restart
is documented in [RESUME_STATUS.md](RESUME_STATUS.md); do not restart this session.

Job **33436622** was submitted with explicit user approval on account `cpsc4880`,
partition `day`, QoS `normal`: 4 CPUs, 24 GB RAM, 4-hour maximum, no GPU.
YCRC's billing policy states standard-tier computation incurs no charges:
https://research.computing.yale.edu/computing-resources/hpc-policies

Live import completed: **5,391 resources**, one patient. Verification passed for
the selected training question. The session reached `awaiting_response` at
2026-11-15T01:09:45.387111+00:00. No teacher action has been submitted.

## Open the fresh teacher conversation

Use this working directory and restrict its file access to this directory:

`/nfs/roberts/project/cpsc4880/cpsc4880_yh774/beyond-triage-fhir/teacher_workspace/sessions/4d772a3230da8ecc5ad9bc61-attempt1`

Give the fresh conversation this prompt:

> Read AGENTS.md and episode.json in this directory. Solve this one question using
> only teacher_client.py to call the provided tools. Record a genuine trajectory
> and stop when the episode ends. Do not inspect files outside this directory.

The teacher can begin with `python3 teacher_client.py status`. Do not copy this
operator conversation into the new context. A changed working directory alone does
not enforce filesystem isolation; configure the session's actual permissions.

Each awaiting-response turn allows 30 minutes. The initial response deadline is
approximately 2026-11-15T01:39:45.387111+00:00. The job also has an overall 4-hour limit.
Check state.json before starting; if it has expired, a new approved run needs fresh
session/output paths. Do not reuse the old attempt.

Operator-only evidence: `logs/teacher-33436622.log` and
`outputs/teacher/4d772a3230da8ecc5ad9bc61-attempt1/live/verification.json`.
Do not expose those paths to the teacher as additional inputs.
