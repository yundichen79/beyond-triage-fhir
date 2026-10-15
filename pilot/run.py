"""Plan, prepare, run, and summarize one zero-shot training-question pilot."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace

from dataset.common import canonical, digest, file_digest, read_questions
from dataset.live import interrupt_cleanup, run_job, verification_output
from dataset.questions import FAMILIES, negative
from pilot.agent import MODEL_CACHE, MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, Settings, TransformersModel, run_agent
from pilot.scoring import score_episode
from pilot.tools import TOOL_DEFINITIONS


ROOT = Path(__file__).resolve().parents[1]


def select_questions(rows, per_family=4, seed=4880):
    if type(per_family) is not int or per_family <= 0:
        raise ValueError("per-family must be a positive integer")
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            groups[row["family"]].append(row)
    if set(groups) != FAMILIES:
        raise ValueError("Training questions must cover all current families")
    selected = []
    for family in sorted(groups):
        group = sorted(groups[family], key=lambda row: digest([seed, "zero-shot-pilot", row["id"]]))
        positives = [r for r in group if not negative(r["ground_truth_answer"], family)]
        negatives = [r for r in group if negative(r["ground_truth_answer"], family)]
        if len(group) < per_family:
            raise ValueError(f"{family} has only {len(group)} training questions; reduce --per-family")
        # One negative where available; the remaining slots favor nonempty cases.
        chosen = negatives[:1] if negatives and per_family >= 2 else []
        chosen += positives[:per_family - len(chosen)]
        if len(chosen) < per_family:
            chosen += [r for r in negatives if r not in chosen][:per_family - len(chosen)]
        selected.extend(chosen)
    return selected


def selection_summary(rows):
    return {"questions": len(rows), "patients": len({r["patient_id"] for r in rows}),
            "split": "train", "complexity": dict(Counter(r["complexity"] for r in rows)),
            "families": {family: {"questions": sum(r["family"] == family for r in rows),
                "negative": sum(r["family"] == family and negative(r["ground_truth_answer"], family) for r in rows)}
                for family in sorted({r["family"] for r in rows})}}


def summarize(records):
    records = list(records)
    def aggregate(group):
        n = len(group)
        correct = sum(r["scores"]["answer_correct"] for r in group)
        grounded = sum(r["scores"]["answer_correct_with_evidence"] for r in group)
        return {"questions": n, "completed": sum(r["status"] == "completed" for r in group),
                "correct": correct, "accuracy": correct / n if n else 0,
                "correct_with_evidence": grounded, "accuracy_with_evidence": grounded / n if n else 0,
                "mean_tool_calls": sum(r["metrics"]["tool_calls"] for r in group) / n if n else 0,
                "input_tokens": sum(r["metrics"]["input_tokens"] for r in group),
                "output_tokens": sum(r["metrics"]["output_tokens"] for r in group),
                "tool_errors": sum(r["metrics"]["tool_errors"] for r in group),
                "protocol_errors": sum(r["metrics"]["protocol_errors"] for r in group)}
    return {"overall": aggregate(records),
            "by_family": {key: aggregate([r for r in records if r["family"] == key]) for key in sorted({r["family"] for r in records})},
            "by_complexity": {key: aggregate([r for r in records if r["complexity"] == key]) for key in sorted({r["complexity"] for r in records})},
            "by_negative": {str(key).lower(): aggregate([r for r in records if r["negative"] == key]) for key in (False, True)},
            "statuses": dict(Counter(r["status"] for r in records))}


def execute_pilot(rows, client, model, settings, output, dataset, replace=False, source_id=None):
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("The initial pilot uses training questions only")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {"type": "run", "started_at": datetime.now(timezone.utc).isoformat(),
              "mode": "zero-shot", "model": model.metadata, "settings": asdict(settings),
              "selection": selection_summary(rows), "question_ids": [r["id"] for r in rows],
              "dataset_sha256": file_digest(Path(dataset) / "questions.jsonl"), "source_id": source_id,
              "fhir_base_url": client.base, "system_prompt": SYSTEM_PROMPT, "tools": TOOL_DEFINITIONS,
              "code_sha256": {str(p.relative_to(ROOT)): file_digest(p) for p in
                  [ROOT / "pilot" / name for name in ("agent.py", "tools.py", "run.py", "scoring.py")]
                  + [ROOT / "dataset/fhir.py"]}}
    light_records = []
    client.strict_search = True
    with output.open("w" if replace else "x") as handle:
        handle.write(canonical(header) + "\n")
        handle.flush()
        try:
            for index, row in enumerate(rows, 1):
                print(f"Pilot {index}/{len(rows)}: {row['family']} ({row['complexity']})", flush=True)
                seed = (settings.seed + int(digest(row["id"])[:8], 16)) % 2**32
                # Do not pass IDs, families, operations, labels, or oracles to the agent.
                episode = run_agent(row["question"], row["patient_id"], model, client, settings, seed)
                result = {"type": "example", "id": row["id"], "patient_id": row["patient_id"],
                          "family": row["family"], "complexity": row["complexity"],
                          "negative": negative(row["ground_truth_answer"], row["family"]), "seed": seed,
                          **episode, "scores": score_episode(row, episode)}
                handle.write(canonical(result) + "\n")
                handle.flush()
                light_records.append({k: result[k] for k in ("id", "family", "complexity", "negative", "status", "metrics", "scores")})
                print(f"  status={result['status']} correct={result['scores']['answer_correct']} "
                      f"tool_calls={result['metrics']['tool_calls']}", flush=True)
        finally:
            summary = {"type": "summary", "finished_at": datetime.now(timezone.utc).isoformat(),
                       "complete": len(light_records) == len(rows), "expected_questions": len(rows),
                       **summarize(light_records)}
            handle.write(canonical(summary) + "\n")
            handle.flush()
            print(json.dumps(summary, indent=2), flush=True)
    return summary


def prepare_model():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError("Install pilot dependencies using bash scripts/prepare_pilot.sh") from exc
    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(MODEL_CACHE))
    print(f"Prepared pinned checkpoint: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "run", "summarize"))
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/cohort-200")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pilot.jsonl")
    parser.add_argument("--replace", action="store_true", help="Replace the single current pilot result file")
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4880)
    parser.add_argument("--model-path", type=Path, help="Optional existing local checkpoint; otherwise use the pinned official model")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-tool-calls", type=int, default=48)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare_model()
            return
        if args.action == "summarize":
            with args.output.open() as handle:
                records = (json.loads(line) for line in handle if line.strip())
                print(json.dumps(summarize(r for r in records if r.get("type") == "example"), indent=2))
            return
        rows = select_questions(read_questions(args.dataset / "questions.jsonl"), args.per_family, args.seed)
        print(json.dumps(selection_summary(rows), indent=2), flush=True)
        if args.action == "plan":
            return
        if args.output.exists() and not args.replace:
            raise ValueError(f"Pilot output exists: {args.output}; use --replace to update the current result")
        if args.output.resolve() == args.dataset.resolve() or args.dataset.resolve() in args.output.resolve().parents:
            raise ValueError("Pilot output must be outside the processed dataset directory")
        settings = Settings(max_steps=args.max_steps, max_tool_calls=args.max_tool_calls,
                            max_context_tokens=args.max_context_tokens, max_new_tokens=args.max_new_tokens,
                            max_output_tokens=args.max_output_tokens, seed=args.seed)
        model = TransformersModel(settings, args.model_path)
        def after_verify(client, selected, snapshot):
            execute_pilot(selected, client, model, settings, args.output, args.dataset,
                          replace=args.replace, source_id=snapshot.id)
        # Reuse verification's isolated HAPI/PostgreSQL lifecycle. Its operational
        # files remain temporary; the only persistent experiment artifact is JSONL.
        with verification_output() as output:
            live_args = SimpleNamespace(output=output, dataset=args.dataset, bundles=None, snapshot=None, batch_size=100)
            with interrupt_cleanup():
                run_job(live_args, rows=rows, after_verify=after_verify)
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
