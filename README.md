# Beyond TRIAGE-FHIR

Supervised fine-tuning of adaptive agents for FHIR clinical question answering.

This project builds on [TRIAGE-FHIR](https://github.com/yundichen79/triage-fhir) by investigating a Qwen3-8B ReAct agent that adapts its reasoning and tool use to each question.

The planned comparison uses the same Qwen3-8B backbone and FHIR environment for a predefined workflow, a zero-shot adaptive agent, and an adaptive agent trained with trajectory-level supervised fine-tuning. Evaluation measures answer correctness, tool interactions, and generated tokens.

## Proposal

See [Beyond TRIAGE-FHIR](docs/Beyond%20TRIAGE-FHIR.pdf).

## Status

Project initialized; implementation is forthcoming.
