# Teacher instructions: one FHIR training episode

You are the teacher for one synthetic patient chart-review question. Solve it with
the real patient tools and submit a structured answer with evidence. The result is
a candidate demonstration, not an automatically approved SFT example.

## Access

Work only inside this session directory and only through `python3 teacher_client.py`.
Do not read `episode.json`, `turn.json` or other files: the client prints everything
you need. Do not inspect parent directories, repository source, raw bundles, labels,
other episodes, operator outputs or prior conversations, and do not access FHIR any
other way. If answer-revealing material is exposed, report it and stop.

## Protocol

1. Run `python3 teacher_client.py status` once. It waits for the backend and prints
   the first turn: the shared `rules`, the `tools`, and the question with its Answer
   JSON Schema. Follow those rules.
2. Submit each action through `respond`, as JSON with `name` and `arguments`, given
   as one single-quoted argument on one line:

   ```bash
   python3 teacher_client.py respond '{"name":"calculate","arguments":{"earlier":"1","later":"2","decimal_places":1}}'
   ```

   (Syntax example only, not a required first action.) Only the two forms
   `python3 teacher_client.py status` and `python3 teacher_client.py respond '<json>'`
   are permitted: no heredocs, pipes, `cd`, `sleep`, `&&` or other commands. Write a
   single quote inside the JSON as `'`. `respond` waits for the
   backend and prints the next turn directly: only the tool results (`new`) added
   since your previous turn, plus turn and student-context counters. Do not run
   `status` after `respond`; only if `respond` reports `submitted`, run `status`
   once. Never resubmit for a turn. Commands can block for a few minutes, so allow
   a long command timeout.
3. Independent calls (for example, reading several referenced records) can go in one
   turn as a JSON array: `respond '[{"name":...},{"name":...}]'`. This saves turns.
   Submit `finish` alone, after observing all necessary results; its arguments are
   `answer` (matching the question's schema) and `evidence`.

Do not invent tool responses or reconstruct a plausible trace afterward. Formatting
errors return public feedback, not answer hints. `--full` prints the whole turn and
`status --repeat` reprints the current turn; use them only if output was lost.

## Solve faithfully

Identify the requested anchor and time window and keep the anchor's identity through
later searches. Do not invent recent-history cutoffs, status filters or clinical
relationships. Follow report result references and explicit indication references.
Match medication products by coded identity. Use requested units and rounding, and
calculate differences from the correct retrieved measurements. Prefer targeted,
filtered searches over broad unfiltered ones: large results consume the student
context budget (`student_context_tokens` of `student_context_available`).

Follow pagination when needed for the requested selection or absence. An error, an
unfinished search or an incorrectly filtered empty result does not establish
absence. Cite retrieved anchor records as well as supporting records. Be efficient
without omitting necessary evidence; do not guess merely to finish within a budget.

## Budgets and termination

The student loop enforces 24 turns, 48 non-finish tool calls, a repeated-failure
guard, and a 16,384-token student transcript (counted with the Qwen tokenizer, with
2,048 tokens reserved per turn). These limits apply to the recorded transcript, not
to this conversation's own context or reasoning.

The episode ends only through `finish` or a budget limit. Do not end your turn
while the episode is still open, even if the budget looks tight. Narrow your searches
(filters, dates, encounter references, smaller pages) and keep going. If a budget runs
out, the backend ends the episode for you.

When the status is `completed`, `step_limit`, `tool_call_limit`, `context_limit`,
`output_token_limit`, `repeated_failure` or `model_error`, the attempt is over:
stop. For `backend_failed`, stop and leave it to the operator. Do not retry outside
the recorded episode, inspect scores, or start another question.
