# Real teacher inputs

The former placeholders have been removed. Prepared, complete inputs for each
training question are in `../public/episodes/<opaque-id>.json`.

Each packet includes `question`, `patient_reference`, `system_prompt` (the exact
shared public v5 rules), `answer_schema`, `tools`, and `budgets`. The launcher copies
one packet to the isolated session as `episode.json`. Do not give the teacher the
original labeled dataset. Use a fresh context per episode.
