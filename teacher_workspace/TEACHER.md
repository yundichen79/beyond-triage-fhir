# Teacher instructions: one FHIR training episode

You are the teacher for one synthetic patient chart-review question. Use actual
patient tools and return a structured answer with evidence. Your interaction is a
candidate demonstration, not an automatically approved SFT example.

## Access and startup

Work only inside this session directory. Read `episode.json` and use
`python3 teacher_client.py status` to obtain the current request. The packet
contains the exact question, patient reference, shared Qwen v5 instructions, one
public answer schema, tool definitions, and budgets. Follow those public rules.
Do not inspect parent directories, repository source, raw bundles, hidden labels,
other episodes, operator outputs, reports, or prior conversations. Do not access
FHIR directly or use another tool to bypass the client. If answer-revealing material
is exposed, report it and stop; the operator must exclude that attempt.

The operator must configure filesystem access restrictions for this session.
These instructions alone cannot enforce isolation. Start a fresh conversation for
this episode; do not inherit the conversation that prepared the experiment.

If status is `starting_backend`, wait for the operator's verified FHIR backend.
If status is `processing`, poll status after a short interval. Do not send another
response until the status is `awaiting_response` with a new turn ID.

## Execute real tools

At each awaiting-response turn, submit one actual assistant response through the
client. Its content is JSON with `name` and `arguments`, using the tool definitions
in the packet. The tools are `search_fhir`, `read_fhir`, `next_page`, `calculate`,
and `finish`. For example, this generic calculator invocation demonstrates syntax
only and is not a required first action:

```bash
python3 teacher_client.py respond <<'JSON'
{"name":"calculate","arguments":{"earlier":"1","later":"2","decimal_places":1}}
JSON
```

Choose your actual action from the question and retrieved records. The backend
executes it through the same student tool implementation. The client's response
may already contain the next turn; otherwise run `python3 teacher_client.py status`.
Each next turn contains the full recorded transcript, including actual tool results.
Do not invent tool responses or reconstruct a plausible successful trace afterward.

Multiple calls in one model turn can be submitted as consecutive `<tool_call>` JSON
blocks, as supported by the student parser. Submit `finish` alone, after observing
all necessary results. Its arguments must contain `answer` and `evidence` and match
the question's schema. Formatting errors produce public feedback, not answer hints.

## Solve faithfully

Identify the requested anchor and time window. Preserve the anchor's identity
through subsequent searches. Do not invent recent-history cutoffs, status filters,
or clinical relationships. Follow report result references and explicit indication
references. Match medication products by coded identity. Use requested units and
rounding; calculate differences from the right retrieved measurements.

Follow pagination when needed for the requested selection or absence. An error,
an unfinished search, or an incorrectly filtered empty result does not establish
absence. Cite retrieved anchor records as well as supporting records. Be efficient
without omitting necessary evidence. Do not guess merely to finish within a budget.

## Budgets and termination

The unchanged student loop enforces 24 submitted model turns, 48 non-finish tool
calls, and the repeated-failure guard. It checks a 16,384-token student transcript,
reserves up to 2,048 output tokens per turn, and caps total submitted output at
16,384 tokens, counted with the pinned Qwen tokenizer. Overlong submitted responses
are truncated by the bridge and the original text is preserved privately.

These are limits on the recorded student-format transcript. They do not measure
or control this Codex conversation's actual context, hidden reasoning, or token
usage. Do not describe them as equal teacher/student compute budgets. Hidden
chain-of-thought is not requested. The client records submitted responses and tool
results, not the complete outer Codex conversation.

When the status is `completed`, `step_limit`, `tool_call_limit`, `context_limit`,
`output_token_limit`, `repeated_failure`, or `model_error`, the attempt is over.
For `backend_failed`, ask the operator to inspect private logs. Do not retry outside
the recorded episode. External scoring stays private and happens after termination.
Stop this context; do not inspect scores or generate another question here.
