# Beyond TRIAGE FHIR context handoff

Prepared October 19, 2026, America/New_York, from the desktop chat.
This is development context, not submission prose or evidence of cluster execution.

## Goal and course constraints

- Yale CPSC 5880 group project by Yundi Chen, Jun Ni, and Yixin Hu.
- Progress report 1 is due Wednesday, October 21, 2026, per the user.
- User clarification: all remaining pre-submission work belongs to Jun. Yundi has crafted the framework and Yixin has drafted the document. There will be no further group collaboration stage before submission.
- Jun wants the remaining work completed today, Monday, October 19, because of limited availability. Treat today as the working deadline; do not defer execution or review to Tuesday or Wednesday.
- The course requires a one-page report using the staff template. Canvas is authoritative for deadline details and the template.
- Course logistics: https://nlp.cs.yale.edu/cpsc4880/logistics/
- AI tools may assist coding, debugging, and experimental analysis. Students must write the proposal, progress reports, and final report themselves; only limited sentence-level polishing is permitted. Disclose AI use.

## Research plan

Compare fixed TRIAGE-FHIR orchestration, a zero-shot adaptive Qwen3-8B agent, and a trajectory-SFT adaptive agent using the same FHIR environment. Evaluate answer correctness and inference effort. The proposed SFT selection favors the shortest successful trajectory per training question, subject to evidence checks.

Repository: https://github.com/yundichen79/beyond-triage-fhir

## State observed by the desktop agent

The desktop agent read the proposal, local progress draft, GitHub README, and selected implementation files on October 19. It did not access Bouchet, run the implementation, or inspect actual experiment logs. The following may already be stale; reconcile with the working tree, active Slurm jobs, and artifacts first.

- Dataset reported: 200 Synthea v4.0.0 patients, FHIR R4, 1,202 questions across 14 families.
- Patient-level split: train 151 patients / 899 questions; validation 22 / 145; test 27 / 158.
- Provisional difficulty counts: 499 simple, 548 intermediate, 155 complex; 90 missing-record cases.
- README reports local validation, raw-source replay, and clinical review passed. These claims were not independently verified by the desktop agent.
- README says live verification awaits a rerun after empty patient-search results. A HAPI configuration workaround and early search probe are implemented; success was not yet established.
- Zero-shot pilot is implemented; README says model preparation and real GPU execution are pending.
- Pilot selection: 56 training questions from 44 patients, four per family, including eight missing-record cases. Difficulty counts in draft: 16 simple, 24 intermediate, 16 complex.
- Pilot configuration in draft: Qwen3-8B, thinking disabled, temperature 0.7, native tool calls, 24 steps, 48 tool calls, 16K context.
- SFT and comparative evaluation remain planned according to the README.

## Recommended checkpoint priorities

The user wants the existing framework and draft brought to a submission-ready state today. The implementation steps below are recommendations for that objective; they do not authorize unrelated extensions.

1. Inspect current code, logs, completed runs, and queued/running jobs. Preserve uncommitted changes and avoid duplicate submissions.
2. Establish live FHIR verification success, or identify and document the precise failing request and expected versus observed behavior.
3. Prepare the pinned model and execute the existing 56-question pilot when prerequisites pass.
4. Preserve provenance and examine representative successes, failures, and missing-record trajectories.
5. Return factual findings and identify inconsistencies in Yixin's existing draft for Jun to resolve today. Do not require a group meeting, teammate review, or another collaboration stage. Dataset expansion, baseline porting, new tools, and SFT are outside today's recommended scope unless needed to fix a demonstrated blocker in the existing pilot.

## Today's completion criteria

- Current live verification outcome is established from actual artifacts, with checked patient/question totals or a precise documented blocker.
- The existing pilot is run to completion if feasible. If blocked, use bounded diagnosis: do not spend the entire day on speculative infrastructure changes. Preserve the partial run and explain exactly what did and did not finish.
- Every available reported metric traces to a run and artifact. Missing results stay explicitly unavailable; incomplete runs are labeled with the actual completed count.
- Provide Jun a compact factual summary, the results table data, and a checklist of claims/placeholders in the existing draft that need correction. Students must write substantive report text themselves under the course policy.
- Check the staff template, one-page requirement, contribution statement, AI disclosure, and exported submission file if those materials are available. Do not assume that producing metrics alone finishes the submission preparation.
- Do not wait for Yundi or Yixin, start optional experiments, or submit course work without Jun's explicit instruction.

## Existing commands and artifacts

Run from the repository root, after checking current README and applicable AGENTS.md instructions. These commands were inspected, not executed by the desktop agent.

```bash
python3 -m unittest discover -s tests -q
python3 -m pilot.run plan
sbatch scripts/verify_live_fhir.sbatch --output outputs/live-report1-attempt1
bash scripts/prepare_pilot.sh
# After prerequisite verification succeeds:
sbatch scripts/run_pilot.sbatch
.venv/bin/python -m pilot.run summarize
```

Choose a fresh explicit live-verification output directory for each attempt; the runner rejects existing directories. The pilot similarly refuses an existing output unless replacement is explicitly requested. Preserve prior experimental evidence.

- Verification launcher: 4 CPUs, 24 GB RAM, 8-hour limit; Java 21 and PostgreSQL modules.
- Pilot launcher: gpu partition, one L40S, 8 CPUs, 64 GB host RAM, 8-hour limit.
- Pilot output: outputs/pilot.jsonl; Slurm log: logs/pilot-<job-id>.log.
- Live-verification default diagnostic files are temporary. An explicit --output retains artifacts.

## Interpretation checks

- Separate implemented functionality from checks actually completed. Do not infer successful live execution from offline validation.
- Pilot is deliberately selected from training questions and balanced by family. Its aggregate accuracy does not estimate held-out performance.
- Confirm all 56 expected episodes and the final JSONL summary's complete=true before reporting a complete pilot. The summarize command aggregates available recorded episodes even for partial runs.
- Summary tool calls are means; output-token counts are totals. Divide by the relevant episode count if presenting mean output tokens.
- Missing-record cases overlap the difficulty groups; do not add them to the denominator.
- answer_correct_with_evidence checks answer correctness plus required-resource retrieval and citation coverage. Coverage does not prove complete search or justify absence claims. Review such trajectories before SFT selection.
- Any policy or evaluator changes should be versioned and documented; retain the frozen test split.

## Return context for the desktop review

Create or update BOUCHET_STATUS.md with the timestamp, branch/commit and dirty state, relevant changes, tests actually run, Slurm job IDs and states, artifact paths, verified totals, pilot completeness and metrics, representative failure observations, blockers, and next action. Clearly distinguish measured results from plans and hypotheses. Include enough command and artifact references to reproduce each claim.

The user can copy that status file and selected logs back to the desktop workspace. Neither agent should assume the other has automatically received updates.
