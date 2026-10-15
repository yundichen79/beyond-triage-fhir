# Beyond TRIAGE-FHIR

Adaptive question answering over FHIR electronic health records, extending
[TRIAGE-FHIR](https://github.com/yundichen79/triage-fhir).

The planned comparison evaluates a fixed workflow, a zero-shot Qwen3-8B ReAct agent,
and an agent trained with trajectory supervised fine-tuning (SFT). We measure
answer accuracy alongside tool calls and token usage.

[Proposal](docs/Beyond%20TRIAGE-FHIR.pdf) · [Question catalog](data/README.md) · [Dataset guide](docs/DATASET.md) · [Current report](docs/COHORT_200_REPORT.md)

## Current status

| Component | Status |
| --- | --- |
| Dataset | Built; local validation, source replay, and clinical review passed |
| Live FHIR verification | Pending rerun after empty patient-search results; a configuration workaround and early search probe are implemented |
| Zero-shot pilot | Implemented; model preparation and actual GPU execution pending |
| SFT and comparative evaluation | Planned; no real-model results collected yet |

## Data source and dataset

**200 synthetic Massachusetts patients** were generated with Synthea v4.0.0 as
FHIR R4 bundles with complete histories. Seeds and dates are pinned in
[cohort.json](configs/cohort.json); exporter settings are in
[synthea.properties](configs/synthea.properties). Original bundles and their
generation record with checksums are preserved in `data/raw/batch-001/`.

The dataset contains **1,202 questions across 14 clinical chart-review families**,
using an exclusive cutoff of **2026-10-02 UTC**. Tasks cover encounter-linked
conditions/orders, CBC and blood pressure, documented indications, medication
comparisons, and longitudinal follow-up. Difficulty is provisional:
499 simple, 548 intermediate, and 155 complex.

| Split | Patients | Questions |
| --- | ---: | ---: |
| Train | 151 | 899 |
| Validation | 22 | 145 |
| Test | 27 | 158 |

All questions for a patient stay in the same split.

## Dataset construction

1. Preserve Synthea bundles and normalize references in a temporary SQLite index.
2. Generate eligible questions and deterministic Python answers, with explicit time
   windows, units, supporting references, and sampled missing-record cases.
3. Validate answers, evidence, question rules, and patient splits.
4. Replay answers from raw bundles and conduct clinical review. Replay uses the
   same answer rules; clinical review provides a separate check.

Only two processed files are kept under `data/processed/cohort-200/`:

| File | Purpose |
| --- | --- |
| `questions.jsonl` | Questions, structured ground truth, evidence references, and evaluator rules |
| `patient_splits.json` | Patient-to-split assignments |

### Build and validate

Run from the repository root. Dataset commands use the Python standard library;
Slurm launchers target Yale Bouchet. Data and artifacts are excluded from Git;
fresh clones must first follow the [source-generation instructions](docs/DATASET.md#build-and-validate).
With source bundles available:

```bash
python3 -m dataset.questions --output data/processed/cohort-200 --replace
python3 -m dataset.validate --dataset data/processed/cohort-200
```

### Verify against live FHIR

```bash
python3 -m dataset.live prepare
sbatch scripts/verify_live_fhir.sbatch
```

Preparation requires Apptainer. The job uses Java 21 and PostgreSQL 16.4 to import
complete charts, check pagination, and recompute answers through HTTP. Failed
requests remain errors. Indexes, servers, and operational artifacts are temporary;
logs remain in `logs/`.

## Zero-shot pilot

The pilot selects **56 training questions from 44 patients**: four per family,
including eight missing-record cases. Validation and test data are reserved;
this pilot checks agent behavior rather than estimating held-out performance.

Qwen3-8B uses native tool calling with thinking disabled. Inputs contain the question,
patient reference, shared instructions, and answer schema. Tools provide patient-scoped
search, reference reads, pagination, and answer submission. Ground truth stays outside
the model.

```bash
python3 -m pilot.run plan        # Preview selection without a model download
bash scripts/prepare_pilot.sh   # Install dependencies and cache the pinned model
```

After live verification passes, submit the GPU run:

```bash
sbatch scripts/run_pilot.sbatch
```

The launcher requests one L40S GPU, eight CPU cores, and 64 GB host memory. It
starts isolated FHIR servers and verifies selected charts before answering.
See [requirements-pilot.txt](requirements-pilot.txt) for pinned dependencies.

One experiment file, **`outputs/pilot.jsonl`**, contains run settings, episode
transcripts, answers, scores, and a summary. Logs go to `logs/pilot-<job-id>.log`.
After the job finishes:

```bash
python3 -m pilot.run summarize
```

Add `--replace` to intentionally replace an existing pilot result.

## Scoring

- **Answers:** compare structured outputs deterministically, matching coded concepts,
  equivalent timestamps, supported unit aliases, and unordered lists with multiplicity.
  Numeric tolerance is `0.000001` in requested units. Missing records and missing
  values remain distinct.
- **Evidence:** report required resource retrieval and citation coverage separately
  from answer accuracy. Coverage alone does not prove a complete search or justified
  absence claim; review transcripts before selecting SFT trajectories.
- **Effort:** record tool calls, HTTP requests, input/output tokens, elapsed time,
  and errors. Context, step, and token limits produce explicit failure statuses.

## Repository layout

```text
configs/       Synthea, cohort, and HAPI settings
dataset/       Generation, answer rules, validation, and live verification
pilot/         Agent loop, FHIR tools, scoring, and runner
scripts/       Preparation and Slurm launchers
tests/         Dataset, FHIR, and agent regression tests
docs/          Proposal, dataset guide, and current report
data/          Preserved raw cohort and processed dataset
vendor/        Pinned artifacts and cached model
outputs/       Current pilot result
logs/          Slurm logs
```

Run offline checks:

```bash
python3 -m unittest discover -s tests -q
```

Next steps: confirm live retrieval, inspect real pilot trajectories, then collect
correct, efficient trajectories for SFT and expand the cohort for evaluation.
