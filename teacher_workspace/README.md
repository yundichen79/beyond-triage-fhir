# Interactive Codex teacher setup

For the full training dataset, use [BATCH.md](BATCH.md). The batch runner starts a
fresh teacher automatically for each remaining question. The instructions below
describe the original single-question interactive pilot.

The public input exporter, file-based tool client, live FHIR launcher, transcript
recorder, and private scorer are implemented. No model API or GPU is required for
the teacher bridge. Running the FHIR backend still requires an approved CPU job.

## Prepared inputs

`public/index.json` lists 899 opaque training episode IDs. Each file under
`public/episodes/` contains only the question, patient reference, exact shared v5
system instructions, question-specific JSON schema, bound tool definitions, and
budgets. No validation/test questions or answer/evidence labels are exported.

The current public wording and v5 interface are preserved, including previously
identified ambiguities; no benchmark or scorer change is hidden in this setup.
If rules are later clarified, version and rerun comparable student evaluations.

## Start one episode (operator, from repository root)

Choose an opaque ID from `public/index.json`. Use new session/output paths for each
attempt. After the user approves a CPU job and the operator verifies its account,
partition, QoS, and billing eligibility, submit:

```bash
sbatch scripts/run_teacher.sbatch \
  --episode EPISODE_ID \
  --session teacher_workspace/sessions/EPISODE_ID-attempt1 \
  --output outputs/teacher/EPISODE_ID-attempt1
```

`EPISODE_ID` is the chosen actual ID, not a literal argument. The launcher requests
4 CPUs, 24 GB RAM, and 4 hours. It does not request a GPU, install packages, download
a model, or call a paid API. It does not assume that any scheduler configuration
is free. Verify eligibility before submission. The backend uses the cached Qwen
tokenizer, Java 21, PostgreSQL, and pinned HAPI artifacts already used by the pilot.
It verifies the selected patient's chart before accepting teacher actions.

A concrete first-pilot command is saved in `FIRST_EPISODE.md` after preparation.
No job is submitted by the exporter or by opening this document.

## Open the fresh conversation

Once the job creates the session and its state becomes `awaiting_response`, open a
fresh Codex conversation **in that session directory**, with no inherited history.
Give it this request:

> Follow AGENTS.md in this directory. Solve this one question using only
> teacher_client.py, starting with `python3 teacher_client.py status`. Record a
> genuine trajectory and stop when the episode ends. Do not inspect any files
> outside this directory.

The client prints compact turns: the first `status` shows the shared rules, tool
definitions and question once; each `respond` waits for the backend and shows only
the new tool results. The teacher does not need to read `episode.json`. Batch runs
pass `TEACHER.md` as the launch prompt and do not copy it into the session.

Configure actual filesystem permissions to deny access to the parent repository,
other sessions, labels, logs, and private outputs. A new working directory or an
AGENTS.md alone does not impose that boundary. This repository cannot reconfigure
the host application's permission profile. If isolation cannot be enforced, record
that limitation and do not claim strict access isolation for those demonstrations.

The client communicates through shared session files, so the Codex process and
CPU job may run on different cluster nodes without SSH tunnels or an exposed HTTP
service. The teacher needs only Python's standard library and session-directory
access. It must not receive the backend repository or operator credentials.

## What is enforced and recorded

The backend reuses `pilot.agent.run_agent` and `PatientTools` unchanged: exact tool
behavior, finish validation, reference scope, pagination, tool/step limits, and
repeated-failure policy. It runs the real HAPI backend, not a fixture emulator.

Token limits are enforced on the student-format transcript using the pinned Qwen
tokenizer. Codex's outer conversation, hidden reasoning, actual token usage, model
identity, and sampling settings are not observable/controllable by this bridge.
Record the visible teacher model identity and settings in private operator notes.
This method produces demonstrations; it is not an equal-compute teacher benchmark.

Private outputs contain `provenance.json`, `raw-turns.jsonl`, `trajectory.json`,
`evaluation.json`, and live verification logs. Scores never enter the public session.
`approved_for_sft` remains false until review of anchor selection, unsupported
filters, and absence completeness. Preserve failures and all intermediate actions.

## Preparation and checks

To export a new version into a new directory:

```bash
python3 -m teacher.prepare --output teacher_workspace/public-next
```

The exporter refuses existing paths. Run backend/client regression checks with:

```bash
python3 -m unittest discover -s tests -p 'test_teacher.py' -v
```

These tests check actual bridge-to-agent execution with fixture transport. They do
not claim a new live HAPI verification or generated teacher demonstration. The live
launcher performs verification when the approved job starts.
