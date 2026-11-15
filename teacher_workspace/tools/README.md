# Callable teacher tools

The implementation is `teacher/client.py` plus `teacher/bridge.py` and
`teacher/serve.py` on the trusted backend. The launcher copies only the public
client into the session as `teacher_client.py`. Tool definitions are bound to each
question's schema in its exported public packet.

The teacher uses `python3 teacher_client.py status` and
`python3 teacher_client.py respond` (JSON on stdin). The backend dispatches the
response through the unchanged `pilot.agent.run_agent` loop and `PatientTools`.
Real HAPI retrieval, observed-reference restrictions, pagination, decimal
calculation, finish validation, and budget counters are shared with Qwen.

The FHIR backend must be running in an approved CPU allocation. A tool definition
is not a running server; see the operator README for the exact launch procedure.
