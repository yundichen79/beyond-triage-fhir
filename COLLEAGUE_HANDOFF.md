# Project handoff — current status and next steps

This note summarizes the latest completed v5 diagnostic and dataset audit. Older README/status notes may predate these results. No new job or Git operation was performed for this handoff.

## Aim and claim

**Beyond TRIAGE-FHIR:** fine-tune Qwen3-8B on verified successful tool-use trajectories for FHIR chart-review questions. Compare a fixed TRIAGE-FHIR baseline, the zero-shot agent, and the SFT agent using the same dataset, tools and budgets. Intended claim: SFT improves accuracy and/or reduces retrieval effort on unseen synthetic patients. Shortest-trajectory selection alone does not demonstrate adaptive reasoning; measure effort by task difficulty. Include a shortest-versus-random verified-trajectory ablation.

## What is complete

- **Dataset:** 200 Synthea v4.0.0 patients, FHIR R4; 1,202 questions across 14 families. Patient splits: 151/22/27; question splits: 899/145/158 train/validation/test. Structured answers and evidence IDs; 90 primary negative cases. Labels describe synthetic record review, not clinical recommendations.
- **Infrastructure:** isolated HAPI FHIR + PostgreSQL Slurm workflow and live label verification are implemented. Earlier full live verification passed; the latest dataset audit was offline and did not rerun it.
- **Agent v5:** Qwen3-8B, thinking disabled; patient-scoped search, retrieved-reference reads, pagination, calculator and validated finish. V5 added decimal arithmetic, rejected contradictory filters and revised retrieval instructions. The prior interface test run passed 97 tests.
- **Completed v5 diagnostic:** job `33428950`, A40, about 20 minutes. **3/14 correct and grounded (21.4%)**, 11 submitted answers, two repeated-failure stops, one context-limit stop. This is one training question per family, with no negative cases—not a full baseline evaluation or the originally proposed 56-question pilot.
- **All 14 traces audited:** eight of eleven failures lacked required retrieved evidence; three had it but constructed wrong answers. Common failures: wrong encounter anchors, invented date cutoffs, incorrect reference associations, malformed tool calls and false absence claims. No new server/scorer defect was confirmed; this does not certify the system as bug-free.
- **Entire dataset audited:** all 1,202 answers and evidence sets matched a separate implementation; canonical raw-source replay passed; 203 source-file checksums matched; no patient or cited-resource overlap across splits. All 41 dataset unit tests passed. Dataset and splits were unchanged.

## Qualifications and unfinished work

- The audit is automated, not clinician review. The separate implementation shares normalization and was written after inspecting the production rules; shared assumptions remain possible.
- Clarify procedure timestamp selection (`performedDateTime`, otherwise `performedPeriod.start`), systolic-based BP encounter eligibility, medication intent/status eligibility, and cutoff conventions. All 186 procedure-indication questions cite a period-represented procedure. Version changes rather than silently editing an evaluated benchmark.
- Coverage is thin: only one glycemic test question, six test negatives overall, and zero missing-numeric-value examples. All families appear in all splits, but family-specific estimates can be unstable. Difficulty tiers are not yet empirically calibrated.
- Offline answerability does not establish feasibility through the agent tools under the context/step limits.
- A stronger teacher has not been selected. Large-scale verified trajectory generation, SFT, the fixed TRIAGE-FHIR comparison, and final held-out comparisons remain unfinished. Do not describe these as completed.

## Next steps, in order

1. **Freeze public task rules.** Resolve the wording points above, version any changes, and rerun the audit. Preserve existing results and splits.
2. **Check tool feasibility on training data.** Implement a scripted reference solver through the exact agent tools and budgets for all 14 families, including negatives, pagination and reference traversal. Use public task rules to retrieve; use hidden labels only for checking. Keep this harness separate from the fixed TRIAGE-FHIR baseline.
3. **Pilot a capable teacher.** Generate trajectories on a small, diverse training subset. Give the teacher only the question, public schema and tools—not expected answers or evidence IDs. Verify correctness and grounding afterward. Measure yield, family coverage, tool calls, tokens and cost before scaling. High zero-shot accuracy is not a prerequisite for SFT; trustworthy demonstrations are.
4. **Build the training corpus.** Sample multiple attempts per training question. Keep the shortest correct, fully grounded trajectory; retain adequate family/negative coverage. Require complete retrieval support for absence claims. Exclude failed attempts and validation/test trajectories. Preserve other verified successes for the random-selection ablation. Use the same tool interface for teacher and student.
5. **Fine-tune and compare.** Train Qwen3-8B; select settings on validation. Freeze configurations before test evaluation. Compare fixed baseline, zero-shot and SFT under equal budgets. Report accuracy, grounded accuracy, family counts/macro accuracy, calls and tokens; use patient-level resampling for uncertainty. Evaluate whether effort varies appropriately with difficulty.

Avoid another large combined prompt/tool/model change without an isolated comparison. Dataset regeneration is not currently supported as the remedy for low baseline accuracy.

## Read these artifacts

- [Dataset audit report](outputs/dataset-audit/REPORT.md), [per-question checks](outputs/dataset-audit/audit.json), [coverage](outputs/dataset-audit/coverage.json).
- [Complete v5 trace audit and debugging proposals](outputs/v5-trace-inspection/all-traces-audit.md).
- [Correct trace](outputs/v5-trace-inspection/correct-cbc-change.md) and [incorrect trace](outputs/v5-trace-inspection/incorrect-glycemic-followup.md).
- [Raw v5 episodes](outputs/pilot-interface-v5-diagnostic.jsonl), [job log](logs/pilot-33428950.log).
- Audit implementation: [answer/evidence checks](scripts/audit_dataset.py), [coverage/source checks](scripts/audit_dataset_coverage.py).

Reproduce the offline audit from the repository root, after obtaining command approval:

```sh
python3 -m dataset.review --dataset data/processed/cohort-200
python3 -m scripts.audit_dataset
python3 -m scripts.audit_dataset_coverage
python3 -m unittest discover -s tests -p 'test_dataset.py'
```

## Permissions and costs

Follow [AGENTS.md](AGENTS.md): inspect only authorized files/directories; obtain approval before commands or changes; never inspect private cluster data. Every Git operation, including status, requires approval. Git tracking/sync status has not been checked here; verify whether linked outputs are tracked before relying on a clone for handoff.

The user requires **no charges for submitted jobs**. Prior approved runs used A40 on standard `gpu`, account `cpsc4880`, QoS `normal`. Verify current billing eligibility and resource configuration before each authorized submission; hardware name alone does not guarantee zero cost. Do not use priority/billable partitions or paid teacher APIs without explicit approval. This handoff authorizes no new submissions or expenditure.
