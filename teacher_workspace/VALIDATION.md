# Setup verification

- Exported and checked all 899 training-only public packets against the allowlisted
  source fields. Validation/test rows are rejected by the exporter.
- Full offline suite: 101 tests passed. After the final journal/client changes,
  all four teacher bridge tests passed again.
- Tests exercise real dispatch through the student agent loop, returned calculator
  values, reference restrictions, step/context limits, stale-response rejection,
  and preservation of actual request/response records.
- Launcher shell syntax passed.
- Pinned local Qwen tokenizer loaded successfully with Torch/TensorFlow disabled;
  a real exported packet rendered through its chat template without downloads.
- Java, PostgreSQL commands, and the HAPI artifact files are present after loading
  the launcher's modules. This is a runtime prerequisite check, not a new live
  verification. Live import/verification is performed by the approved backend job.
- No teacher job, paid API call, or model generation was started by setup.

Activation still requires approval for the prepared CPU job and externally
configured filesystem restrictions on the fresh teacher session. See
`FIRST_EPISODE.md` for the concrete command. Actual Codex context and usage cannot
be enforced or measured by this file bridge; its token counters describe the
Qwen-tokenized submitted transcript.

## Approved live activation

Job 33436622 subsequently imported 5,391 resources and passed live verification
for one patient and one question. The teacher session is awaiting its first response
as of 2026-11-15T01:09:45.387111+00:00. See ACTIVE_SESSION.md for the live handoff.
