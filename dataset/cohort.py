"""Download a pinned Synthea release and generate immutable population batches."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dataset.common import file_digest, write_json
from dataset.bundles import iter_bundle_entries

ROOT = Path(__file__).resolve().parents[1]


def download_jar(config, destination):
    version = config["synthea_version"]
    # Discover the asset by name rather than assuming the release filename.
    request = urllib.request.Request(
        f"https://api.github.com/repos/synthetichealth/synthea/releases/tags/{version}",
        headers={"User-Agent": "beyond-triage-fhir"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    assets = [a for a in release["assets"] if a["name"].endswith(".jar") and "dependencies" in a["name"]]
    if len(assets) != 1:
        raise ValueError(f"Expected one standalone JAR in {version}, found {len(assets)}")
    asset = assets[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"Refusing to replace {destination}")
    tmp = destination.with_suffix(".download")
    try:
        urllib.request.urlretrieve(asset["browser_download_url"], tmp)
        # Recent GitHub releases expose a digest; record ours regardless.
        expected = asset.get("digest")
        actual = "sha256:" + file_digest(tmp)
        if expected and expected != actual:
            raise ValueError("Downloaded JAR digest does not match GitHub release metadata")
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)
    write_json(destination.with_suffix(".provenance.json"), {
        "release": version, "release_id": release["id"],
        "asset_url": asset["browser_download_url"], "sha256": file_digest(destination),
        "github_digest": asset.get("digest"),
    })


def generation_command(config, batch, jar, output, properties, java="java"):
    return [
        java, "-Duser.timezone=UTC", "-Xmx4g", "-jar", str(jar),
        "-s", str(config["batch_seeds"][batch - 1]),
        "-cs", str(config["clinician_seed"]), "-p", str(config["patients_per_batch"]),
        "-r", config["reference_date"], "-e", config["end_date"], "-o", "false",
        "-c", str(properties), f"--exporter.baseDirectory={output}",
        config["state"],
    ]


def generate(config, batch, jar, root, properties):
    java = shutil.which("java")
    if not java:
        raise ValueError("Java is unavailable. On Bouchet run: module load Java/17")
    if not jar.is_file():
        raise ValueError(f"Missing JAR: {jar}. Run the download command first.")
    output = root / f"batch-{batch:03d}"
    if output.exists():
        raise ValueError(f"Refusing to overwrite batch {output}; inspect failed runs separately.")
    # All batches must use the same generator binary and properties.
    for sibling in root.glob("batch-*/generation.json"):
        previous = json.loads(sibling.read_text())
        if previous["jar_sha256"] != file_digest(jar) or previous["properties_sha256"] != file_digest(properties):
            raise ValueError(f"Generator changed since {sibling}; use a new cohort directory.")
        for key in ("reference_date", "end_date", "state", "clinician_seed", "patients_per_batch"):
            if previous["config"][key] != config[key]:
                raise ValueError(f"Cohort setting {key} changed since {sibling}")
        if previous["seed"] == config["batch_seeds"][batch - 1]:
            raise ValueError(f"Seed already used by {sibling}")
    output.mkdir(parents=True)
    command = generation_command(config, batch, jar.resolve(), output.resolve(), properties.resolve(), java)
    environment = {**os.environ, "TZ": "UTC"}
    manifest = {
        "config": config, "batch": batch, "seed": config["batch_seeds"][batch - 1],
        "jar_sha256": file_digest(jar), "properties_sha256": file_digest(properties),
        "properties": properties.read_text(), "command": command,
        "java_version": subprocess.run([java, "-version"], capture_output=True, text=True, check=True).stderr,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
    }
    write_json(output / "generation.json", manifest)
    with (output / "generation.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    manifest["exit_code"] = result.returncode
    patient_ids = []
    for path in sorted((output / "fhir").glob("*.json")):
        patient_ids.extend(e["resource"]["id"] for e in iter_bundle_entries(path)
                           if e.get("resource", {}).get("resourceType") == "Patient")
    manifest["patient_count"] = len(set(patient_ids))
    ok = result.returncode == 0 and len(patient_ids) == len(set(patient_ids)) == config["patients_per_batch"]
    manifest["status"] = "complete" if ok else "failed"
    manifest["files"] = {str(p.relative_to(output)): file_digest(p) for p in sorted(output.rglob("*.json"))
                         if p.name != "generation.json"}
    write_json(output / "generation.json", manifest)
    if not ok:
        raise ValueError(f"Generation failed/count mismatch: see {output / 'generation.log'}")
    print(f"Generated {manifest['patient_count']} patients in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "generate"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cohort.json")
    parser.add_argument("--properties", type=Path, default=ROOT / "configs/synthea.properties")
    parser.add_argument("--jar", type=Path, default=ROOT / "vendor/synthea-v4.0.0.jar")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 1 <= args.batch <= len(config["batch_seeds"]):
        parser.error("--batch is outside configured seed list")
    try:
        if args.action == "download":
            download_jar(config, args.jar)
            print(f"Downloaded {args.jar}; SHA256={file_digest(args.jar)}")
        else:
            generate(config, args.batch, args.jar, args.output, args.properties)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
