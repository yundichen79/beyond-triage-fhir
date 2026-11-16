# Teacher trajectory generation

A teacher model solves training questions through the unchanged student environment
(`student/`), so every recorded trajectory is in the student's format. Each question
runs in its own fresh teacher process: one conversation must never solve several
questions, or later ones would see earlier patients and answers. Validation and test
questions are never selected. This guide and the operator logs are never given to a
teacher.

## How one question runs

1. The batch starts one isolated HAPI FHIR + PostgreSQL backend, imports the selected
   patients' charts and checks every selected answer against the live server.
2. For each question, it creates a public session folder holding only the question
   packet (`episode.json`) and the client (`teacher_client.py`).
3. It starts a fresh teacher process in that folder. The teacher calls
   `python3 teacher_client.py status` and `respond '<json>'`. Each action is executed
   by the real student agent loop and tools against the live backend.
4. The student loop enforces the student budgets: 24 turns, 48 tool calls, the
   repeated-failure guard and the 16,384-token Qwen transcript.
5. After the episode ends, the trajectory is scored privately. The teacher never sees
   scores or labels.

The teacher's instructions are [TEACHER.md](TEACHER.md).

## Teachers

**Claude Code (`--teacher claude`), used for the current corpus.** Each question runs
in a fresh `claude -p` process.
- **Login:** requires a claude.ai subscription login (`claude auth status` must
  report `authMethod: claude.ai`, first-party). The runner removes API-key, auth-token,
  base-URL and Bedrock/Vertex/Foundry settings, so a run cannot switch to per-token
  billing.
- **Model:** a model and effort are required; nothing falls back to the CLI default.
- **Prompt and tools:** `TEACHER.md` replaces Claude Code's system prompt. Bash is the
  only tool, and only `python3 teacher_client.py ...` is allowed; everything else is
  denied without prompting.
- **Isolation:** `--safe-mode` drops the operator's CLAUDE.md, plugins, hooks, skills
  and MCP servers, and nothing is persisted. This is permission-based isolation, not
  an OS sandbox.
- **Events:** written to `attempts/<id>/claude-events.jsonl`.

**Codex (`--teacher codex`, the default).** Each question runs in a fresh
`codex exec` process with an OS-level filesystem sandbox limited to the session folder.
It requires an existing ChatGPT sign-in, and API-key mode is refused.

Claude Code and Codex count cached tokens differently and have different allowances,
so compare token totals with care.

## Launching

Every run needs new `--sessions` and `--output` paths. Sessions (teacher-visible) and
outputs (private) must be separate folders. Jobs use account `cpsc4880`, partition
`day`, QoS `normal`, 4 CPUs, 24 GB RAM and no GPU (free standard tier). The script's
default time limit is 4 hours; long runs pass `--time` and a matching
`--max-run-seconds` (the soft budget, which includes backend setup).

Preview the selection without running anything:

```sh
python3 -m teacher.batch plan --teacher claude --teacher-model claude-sonnet-5-5 \
  --reasoning-effort high --resume-from outputs/teacher-batches/sonnet55-v6-all-004
```

Generate trajectories for all remaining training questions, skipping questions that
already have an evaluated attempt in the listed runs:

```sh
sbatch --parsable --time=23:59:00 scripts/run_teacher_batch.sbatch \
  --teacher claude --teacher-model claude-sonnet-5-5 --reasoning-effort high \
  --resume-from outputs/teacher-batches/sonnet55-high-28-v6 \
  --resume-from outputs/teacher-batches/sonnet55-v6-all-001 \
  --resume-from outputs/teacher-batches/sonnet55-v6-all-002 \
  --resume-from outputs/teacher-batches/sonnet55-v6-all-003 \
  --resume-from outputs/teacher-batches/sonnet55-v6-all-004 \
  --max-run-seconds 84600 \
  --sessions outputs/teacher-sessions/sonnet55-v6-all-005 \
  --output outputs/teacher-batches/sonnet55-v6-all-005
```

Run a frozen selection instead (it cannot be combined with `--resume-from` or `--limit`):

```sh
sbatch --parsable scripts/run_teacher_batch.sbatch \
  --teacher claude --teacher-model claude-sonnet-5-5 --reasoning-effort high \
  --selection configs/selections/sol61-high-28-v1.json \
  --sessions outputs/teacher-sessions/<run> --output outputs/teacher-batches/<run>
```

Resuming re-runs attempts that ended in `model_error`: usage-limit interruptions and
abandoned episodes. Every other evaluated attempt is skipped, whatever its score.
Resume sources must use the same dataset; the runner checks its fingerprint.

## Usage limits and stops

- **5-hour limit (Claude):** the batch keeps the backend running, sleeps until the
  reset time Claude reports (plus 2 minutes), then retries that question in a fresh
  session and folder (`<id>-retry1`, ...).
- **It stops instead** (`stop_reason` in `summary.json`) when:
  - a longer window, such as the 7-day limit, is exhausted
  - the reset falls outside the job's time budget
  - the same question is rejected 3 times
  - the error is not a usage limit (login, permission, other runtime errors)
- **Abandoned episodes:** if Claude ends its own run normally with the episode still
  open, that question counts as a failed attempt and the batch continues. Three in a
  row stop the batch (`repeated_abandoned_episodes`).
- **Time budget:** the batch stops between questions before one would exceed
  `--max-run-seconds`.

To use an early reset of the 5-hour limit, cancel the sleeping job and resubmit with
all runs as resume sources.

## Outputs

Each run's output folder contains:
- `batch.json`: settings, selection, dataset fingerprint and code hashes
- `results.jsonl`: one line per attempt
- `summary.json`: progress counters
- `live/`: backend verification logs
- `attempts/<id>/`: the trajectory, raw request/response journal, provenance, private
  score and teacher event logs

In `summary.json`:
- `attempted` counts questions with a finished attempt, abandoned ones included.
- Interruptions are counted in `interrupted_attempts` and `usage_limit_waits`.
- `complete` means every selected question has a finished attempt, not that every
  answer was correct.

## Collecting the corpus

Combine runs that used the same student interface and dataset into one corpus:

```sh
python3 -m teacher.collect --runs outputs/teacher-batches/sonnet55-high-28-v6 \
  outputs/teacher-batches/sonnet55-v6-all-00{1,2,3,4} \
  --output outputs/teacher-corpus/<name>
```

This writes `trajectories.jsonl` (one line per finished attempt) and `corpus.json`.
`corpus.json` holds the shared system prompt and tools, the sources, the selection
rule and the summary. Interrupted and abandoned attempts are skipped. For each
question, the shortest grounded attempt is marked `selected`.

`selected` is not approval for SFT (`approved_for_sft` stays false). Review anchors,
unsupported filters, references and absence claims before training.

To compare a run with another on the same frozen selection, use
`python3 -m teacher.report --output <run> --baseline <other run>`.

## Exporting public question packets

`python3 -m teacher.prepare` exports the public packet for every training question to
`outputs/teacher-public`. It refuses to replace an existing folder.

## Test coverage

The offline tests use a fixture FHIR transport and never call a teacher model. They
cover:
- training-only selection, dataset checks and resuming
- teacher command construction and the removal of API-key settings
- usage-limit and abandonment handling
- corpus collection
- a complete question through the real agent loop
