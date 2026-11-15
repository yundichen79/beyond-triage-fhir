# Teacher token usage: current problems and required changes

Status as of 2026-11-15. This comes from a read-only investigation of the Teacher
batch runs:

- Previous Teacher: job `33467807`, output in `outputs/teacher-batches/batch-002`
- Sol High: job `33486023`, output in `outputs/teacher-batches/sol61-high-28-v1`

No code was changed and nothing was rerun.

## Summary

Each Teacher question uses about **250K Codex input tokens**. The cause is how the
Teacher talks to the tools through Codex (the bridge protocol), not the FHIR dataset
and not the choice of model. The **context-limit failures** have a separate cause.
They are student-format transcript overflows: a single oversized FHIR search result
fills the 16K-token student budget, often after the needed evidence was already
retrieved.

## Measurements

| Metric | Previous Teacher | Sol High |
|---|---:|---:|
| Mean Codex input tokens per question | 248K | 258K |
| Share of input tokens that were cached | ~89% | ~87% |
| Mean output tokens per question | 1.3K | 1.6K |
| Mean reasoning tokens per question | 26 | 206 |
| Mean Codex shell commands per question | 9.0 | 9.6 |
| Mean time per question | 45 s | 52 s |
| `context_limit` failures | 7 / 63 | 2 / 11 |

Caveats:
- Only 41 of the 74 attempts contain Codex usage events, so the token averages cover
  those 41.
- The Sol job was cancelled (SIGTERM at 15:16) after 11 of its 28 questions.
- `outputs/teacher-batches/sol61-high-28-v1/COMPARISON.md` is stale: it still shows
  0 evaluated. Regenerate it with `python3 -m teacher.report`.

## Problem 1: The Codex bridge re-sends large, duplicated context (main token cost)

Every Codex step re-sends the whole conversation so far. I fitted a model to the 41
attempts with usage events (R² 0.94, mean error about 7%). It splits the ~250K tokens
per question like this:

| Source | Tokens per question | Share |
|---|---:|---:|
| Fixed Codex prefix (~13.9K tokens per step) × ~10 steps | ~142K | 57% |
| Unfiltered `teacher_client.py status` / `respond` JSON output | ~58K | 23% |
| Startup `cat AGENTS.md` + `cat episode.json` | ~30K | 12% |
| Filtered output, command text, ad-hoc inspection | ~21K | 8% |

Causes:

1. **Every turn returns the full transcript, unchanged parts included.**
   - `teacher/client.py` `show()` prints the whole `turn.json` with `indent=2`.
   - `teacher/bridge.py` `FileTeacher.generate()` writes into that file the system
     prompt, all 5 tool definitions, the answer schema and the full message history.
   - The result is 29–52K characters per call, even on the first turn.
2. **The same static content is read three times.**
   - It comes in through `episode.json`, the first `status`, and every `respond`.
   - The answer schema also appears twice inside each copy: once in the user message
     and once in the `finish` tool parameters.
   - `episode.json` alone is 32–48K characters.
3. **There are too many Codex steps, about 10 per question for about 5 student turns.**
   - 50 of 124 unfiltered `status` calls re-read a turn the Teacher had already seen.
   - Two steps are always spent on `cat` startup commands.
   - Every step costs at least ~14K tokens just for the fixed prefix.
4. **Codex probably truncates long command output.**
   - 280 of 684 command outputs are over 10K characters.
   - The fit works best when outputs are assumed to be cut at about 10K characters.
     This is inferred from the fit, not confirmed from Codex source.
   - If it holds, the Teacher may not see the middle of the dumps. That would explain
     why it starts piping output through ad-hoc `python3 -c` filters.
5. **Changing the model cannot fix this.** Sol and the previous Teacher make the same
   number of steps and get the same output sizes.

## Problem 2: Context-limit failures in the student transcript

- The limit is `Settings.max_context_tokens = 16384` Qwen tokens (`pilot/agent.py`),
  checked in `FileTeacher.generate()`. It is not Codex's own context window.
- Each turn reserves 2,048 output tokens, so 14,336 tokens are usable.
- The fixed system prompt, tool definitions and schema already take 3.7–5.9K tokens.
- That leaves only about **8.5–10.5K tokens for all tool results**.

Observed failures:
- **6 of the 7** `context_limit` failures in batch-002 had already retrieved all the
  required evidence (`evidence_recall = 1.0`).
- **5 of the 7** were caused by one fallback search
  `{"resource_type": "Condition", "count": 50}`. It returns about 37.4K characters
  (roughly 10K tokens), which exceeds the remaining budget by itself. This came after
  the Teacher's Condition search filtered by encounter returned nothing. Example:
  attempt `0d4c31a7` went from 4,146 tokens to over the limit after this one call.
- Other cases:
  - CBC questions: 5 DiagnosticReports per page × 2 pages, at about 1.9K characters
    each (`0fe7582e`).
  - Long paging through MedicationRequest results (`0060c0b1`, `0f254712`).
- Both Sol failures are the same questions that failed before, so these failures are
  deterministic.

## Problem 3: Tool results are wordier than needed

`pilot/tools.py` `projection()` already drops narratives and server metadata.
Average sizes are still Condition ≈ 750 characters, DiagnosticReport ≈ 1,850 and
Encounter ≈ 370.

What wastes space:
- `subject` repeats the patient reference in every resource. That is about 14% of
  resource characters, even though patient scope is automatic.
- `clinicalStatus` and `verificationStatus` carry full code-system URLs.
- `code.text` duplicates `code.coding[].display`.
- DiagnosticReport `result[]` lists 13–16 references, each with a long `display`.
  DiagnosticReport also includes `issued`.
- `search_fhir` allows `count` up to 50, and pages are not limited by size.
- The `search_fhir` tool description alone is about 6.1K characters, sent with every
  student turn.

## Problem 4: Dataset (minor, and not a token cause)

- **The 90,082 resources are full charts.** They are 25 complete patient charts plus
  the records they reference (`live/scope.json`). They are imported once, which took
  about 6 minutes. The model never sees the database size, only query results.
- **28% of imported resources can never be queried.** That is 24,929 records of types
  no tool can reach: Claim/ExplanationOfBenefit 16,238, DocumentReference 5,030,
  Immunization 1,459, MedicationAdministration 1,065 and smaller types. They cost
  import time only.
- **Full charts of the reachable types are required.** Questions ask about
  "complete history" and about records being absent. Do not trim them.
- **Synthea's noisy Conditions inflate unfiltered searches.** In the 5 oversized
  Condition pages, 143 of 250 entries were "situation" or "finding" conditions (for
  example "Medication review due"), and 139 repeated a code already on the page.
  Handle this in retrieval (see items 5 and 8 below), not by deleting data, because
  deleting it changes the questions' answers.

## What to modify (in priority order)

### A. Teacher harness: biggest savings; the student data format is unchanged

1. **`teacher/client.py`: return only what changed.**
   - Make `status`/`respond` print only `turn_id`, `status` and the tool results
     added since the previous turn, as compact JSON.
   - Keep the full dump behind an optional `--full` flag.
   - Expected effect: removes most of the 23% bucket and avoids output truncation.
2. **Compact startup.**
   - In `teacher/batch.py` (`PROMPT`, `attempt()`), put the essential instructions
     into the launch prompt.
   - Give the Teacher one compact brief: question, answer schema once, short tool
     list.
   - Stop telling it to `cat AGENTS.md` and `episode.json` (`teacher_workspace/TEACHER.md`).
   - Expected effect: removes most of the 12% bucket and 2 steps.
3. **Fewer Codex steps.**
   - Make `respond` block until the next turn is ready. It currently waits up to 30 s
     and then tells the Teacher to poll.
   - Add an instruction: never re-run `status` for a turn already seen.
   - In `TEACHER.md`, encourage several independent tool calls in one turn as
     multiple `<tool_call>` blocks. The parser already supports this.
   - Goal: about 5 steps instead of about 10, which roughly halves both tokens and
     time.
4. **Optional, larger change:** call the Teacher model through the API with native
   tool calling instead of `codex exec` plus shell commands. This removes the ~14K
   fixed Codex prefix on every step.

### B. Student interface: fixes the context limits

Change these consistently for the student and the Teacher, and bump
`INTERFACE_VERSION` (`pilot/contract.py`), because they change the SFT data format.

5. **Limit page size (`pilot/tools.py`).**
   - Lower the maximum `count` (for example to 20), or stop a page at about 2.5K
     tokens and return `next_page`.
   - Pagination keeps evidence complete.
6. **Slim `projection()`.**
   - Drop `subject`.
   - Reduce `clinicalStatus`/`verificationStatus` to the bare code.
   - Drop `code.text` when it equals `display`.
   - Drop DiagnosticReport `issued` and the `display` on `result[]` references.
   - Expected effect: about half the size per resource. Keep every `id` and
     reference, because citations and scoring depend on them.
   - First check that `pilot/scoring.py` and the system-prompt rules don't rely on a
     removed field. For example, the prompt mentions `issued` for `report_time`.
7. **Remove fixed duplication (`pilot/agent.py`, `pilot/tools.py`).**
   - Send the answer schema once, not in both the user message and the `finish`
     tool.
   - Shorten the `search_fhir` description.
   - This frees about 1–3K student tokens on every turn.
8. **Give a better path for inpatient-condition questions** than an unfiltered
   Condition dump. One option: guidance to read `Encounter.diagnosis` /
   `reasonReference` targets, or to filter by encounter and date. Prompt text alone
   is not enough; it needs the page cap from item 5.

### C. Dataset / setup

9. **`dataset/scope.py` / `dataset/live.py`: filter types at import.** Import only
   the types the tools can reach (Patient, Encounter, Condition, Observation,
   DiagnosticReport, MedicationRequest, Medication, Procedure) plus their reference
   closure. This saves about 28% of import time and changes no evidence. Confirm
   first that the verification and oracle steps don't need the other types.

## Verifying the changes

- Rerun the offline test suite. Some tests check the current interface (projection
  fields, schemas) and will need updating.
- Run a small, approved Teacher batch on the same frozen selection
  (`teacher_workspace/selections/sol61-high-28-v1.json`) and compare against the
  current runs:
  - Codex `input_tokens` per question, from `turn.completed` in `codex-events.jsonl`
  - Codex steps per question
  - Time per question
  - `context_limit` count
  - `answer_correct_with_evidence`
- Target: under 120K input tokens per question, with no context limits on the 7
  known failing questions.
