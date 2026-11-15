# Recorded teacher trajectories

The backend saves real interactions to the private operator output directory:

- `provenance.json`: dataset and code fingerprints, settings and accounting limits.
- `raw-turns.jsonl`: every accepted transport response, original text, truncation,
  and Qwen-tokenized input/output counts.
- `trajectory.json`: actual student-format messages and tool results, final answer,
  citations, retrieved resources, termination and counters.
- `evaluation.json`: post-episode scores, never exposed to the teacher.

Codex's outer chat and hidden reasoning are not captured. These files record the
submitted assistant responses and actual tool execution; do not label token counts
as actual Codex usage. Private output must remain outside the teacher's permissions.

`approved_for_sft` defaults to false even when scoring passes. Review task anchors,
search restrictions, reference links, and completeness of absence claims. Preserve
all intermediate failures rather than editing them out of a successful trace.
Select verified successes only from training patients, maintain family/negative
coverage, and preserve alternative successes for the selection-strategy ablation.
