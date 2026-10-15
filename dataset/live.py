"""Prepare a pinned HAPI artifact, then run an isolated live cohort verification job."""

import argparse
import sqlite3
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from pathlib import Path

from dataset.common import SQLiteResources, Snapshot, file_digest, read_questions, write_json
from dataset.fhir import FHIR, check_capability, import_snapshot, verify_dataset
from dataset.scope import scope_for_patients
from dataset.validate import validate
from dataset.snapshot import add_source_arguments, open_records


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ("docker://hapiproject/hapi@sha256:"
         "34c86fd5805df77c2b9d9c10538050b16ac3dc244352da0ebe4717f931330775")
ARTIFACT = ROOT / "vendor" / "hapi-v8.8.0-1"


def now():
    return datetime.now(timezone.utc).isoformat()


def war_launcher(path):
    # Some images contain multiple generations of loader classes. Only the
    # manifest identifies the launcher that matches the packaged application.
    with zipfile.ZipFile(path) as archive:
        entries = set(archive.namelist())
        if "META-INF/MANIFEST.MF" not in entries:
            raise ValueError("HAPI WAR executable manifest is missing")
        unfolded = archive.read("META-INF/MANIFEST.MF").decode("utf-8").replace("\r\n", "\n").replace("\n ", "")
        manifest = {}
        for line in unfolded.splitlines():
            if not line:
                break
            key, separator, value = line.partition(": ")
            if separator:
                manifest[key] = value
        launcher = manifest.get("Main-Class")
        if launcher not in {"org.springframework.boot.loader.WarLauncher",
                            "org.springframework.boot.loader.launch.WarLauncher"}:
            raise ValueError(f"Unsupported HAPI WAR Main-Class: {launcher}")
        if launcher.replace(".", "/") + ".class" not in entries:
            raise ValueError("HAPI WAR declared launcher class is missing")
        start_class = manifest.get("Start-Class", "")
        if not start_class or "WEB-INF/classes/" + start_class.replace(".", "/") + ".class" not in entries:
            raise ValueError("HAPI WAR declared application class is missing")
        return launcher


def artifact_info(directory=ARTIFACT):
    directory = Path(directory)
    info = json.loads((directory / "artifact.json").read_text())
    war = directory / "main.war"
    if info["image"] != IMAGE or info["war_sha256"] != file_digest(war):
        raise ValueError("Pinned HAPI artifact identity/checksum differs; prepare a fresh artifact")
    # Retain old preparation provenance while deriving execution from the WAR.
    launcher = war_launcher(war)
    return war, dict(info, prepared_launcher=info.get("launcher"), launcher=launcher,
                     launch_mode="java -jar (WAR manifest)")


def prepare():
    if ARTIFACT.exists():
        war, _ = artifact_info()
        print(f"Pinned HAPI artifact already verified: {war}", flush=True)
        return
    apptainer = shutil.which("apptainer")
    if not apptainer:
        raise ValueError("apptainer is required to extract the official HAPI image")
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".hapi-download-", dir=ARTIFACT.parent))
    try:
        # Extraction only: no Docker daemon, container shell, or root execution.
        env = dict(os.environ, APPTAINER_TMPDIR=str(temporary),
                   APPTAINER_CACHEDIR=str(temporary / "cache"))
        subprocess.run([apptainer, "build", "--disable-cache", "--sandbox", str(temporary / "rootfs"), IMAGE],
                       check=True, env=env)
        war = temporary / "rootfs" / "app" / "main.war"
        info = {"image": IMAGE, "image_tag": "v8.8.0-1", "war_sha256": file_digest(war),
                "launcher": war_launcher(war), "prepared_at": now(),
                "apptainer_version": subprocess.check_output([apptainer, "--version"], text=True).strip()}
        # Keep only the executable and its provenance, not the extracted OS image.
        package = temporary / "package"
        package.mkdir()
        war.rename(package / "main.war")
        write_json(package / "artifact.json", info)
        package.rename(ARTIFACT)
        print(f"HAPI prepared: {ARTIFACT}", flush=True)
    finally:
        if temporary.exists():
            # Extracted OCI directories can be read-only. Only adjust our temporary tree.
            for parent, directories, _ in os.walk(temporary):
                for directory in [Path(parent)] + [Path(parent) / d for d in directories]:
                    if not directory.is_symlink():
                        directory.chmod(directory.stat().st_mode | 0o700)
            shutil.rmtree(temporary)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(client, process, timeout=900):
    deadline = time.monotonic() + timeout
    last_error = "not yet reachable"
    next_notice = time.monotonic()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ValueError(f"HAPI exited before readiness (exit {process.returncode}); see hapi.log")
        try:
            return check_capability(client)
        except (OSError, ValueError) as exc:
            last_error = str(exc)
        if time.monotonic() >= next_notice:
            print(f"Waiting for HAPI R4 metadata: {last_error}", flush=True)
            next_notice = time.monotonic() + 30
        time.sleep(5)
    raise ValueError(f"HAPI startup timed out: {last_error}; see hapi.log")


def stop_java(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def paging_preflight_patient(snapshot, rows, boundary=2200):
    """Test the largest selected chart before spending time on the full import."""
    patients = {row["patient_id"] for row in rows}
    if isinstance(snapshot.resources, SQLiteResources):
        counts = snapshot.resources.connection.execute(
            "SELECT patient,count(*) FROM resources WHERE resource_type='Observation' "
            "GROUP BY patient HAVING count(*) > ?", (boundary,))
    else:
        counts = ((patient, len(snapshot.for_patient(patient, "Observation"))) for patient in patients)
    candidates = [(count, patient) for patient, count in counts if patient in patients and count > boundary]
    return max(candidates)[1] if candidates else None


@contextmanager
def interrupt_cleanup():
    def interrupted(number, frame):
        signal.signal(number, signal.SIG_IGN)
        raise InterruptedError(f"Verification job received signal {number}")
    previous = {number: signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def run_job(args, *, rows=None, after_verify=None):
    """Own the live server lifecycle; an optional pilot runs before cleanup."""
    # This command owns its output and temporary database; refuse accidental reruns.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "starting", "passed": False, "started_at": now(),
              "output": str(output), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "hostname": socket.gethostname(), "python_version": sys.version,
              "server_lifetime": "This job only"}
    write_json(output / "status.json", status)
    temporary, process, pg_started = None, None, False
    sources = ExitStack()
    error = None
    try:
        for command in ("java", "initdb", "pg_ctl", "createdb"):
            if not shutil.which(command):
                raise ValueError(f"Missing {command}; load Java/21 and PostgreSQL/16.4-GCCcore-13.3.0")
        war, artifact = artifact_info()
        all_rows = read_questions(args.dataset / "questions.jsonl")
        if rows is None:
            rows = all_rows
        else:
            by_id = {row["id"]: row for row in all_rows}
            if not rows or any(row != by_id.get(row.get("id")) for row in rows):
                raise ValueError("Selected questions must be unchanged rows from the current dataset")
            rows = list(rows)
        status.update({"dataset": str(args.dataset.resolve()), "questions_sha256": file_digest(
            args.dataset / "questions.jsonl"), "hapi_artifact": artifact,
            "config_sha256": file_digest(ROOT / "configs" / "hapi.yaml"),
            "code_sha256": {name: file_digest(ROOT / "dataset" / name) for name in
                            ("live.py", "fhir.py", "scope.py", "common.py", "oracle.py", "validate.py")},
            "java_version": subprocess.run(["java", "-version"], capture_output=True, text=True,
                                           check=True).stderr.strip(),
            "postgres_version": subprocess.check_output(["pg_ctl", "--version"], text=True).strip()})
        print("Loading and checking source records", flush=True)
        snapshot = sources.enter_context(open_records(getattr(args, "bundles", None), getattr(args, "snapshot", None)))
        # Reject altered labels/metadata before loading a server.
        local = validate(snapshot, args.dataset)
        if not local["valid"]:
            raise ValueError("Local dataset validation failed")
        print("Selecting patient charts and referenced resources", flush=True)
        scoped = scope_for_patients(snapshot, {row["patient_id"] for row in rows})
        write_json(output / "scope.json", scoped.scope)
        status.update({"snapshot_id": snapshot.id, "patients": len(scoped.patients),
                       "questions": len(rows), "resources": len(scoped.resources)})
        print(f"Import scope: {len(scoped.patients)} patients, {len(scoped.resources)} resources", flush=True)
        temporary = Path(tempfile.mkdtemp(prefix="beyond-fhir-", dir=os.environ.get("SLURM_TMPDIR") or None))
        print("Staging the verified HAPI WAR in job temporary storage", flush=True)
        staged_war = temporary / "main.war"
        shutil.copyfile(war, staged_war)
        if file_digest(staged_war) != artifact["war_sha256"]:
            raise ValueError("Staged HAPI WAR checksum differs from the prepared artifact")
        pgdata = temporary / "pgdata"
        pgport, httpport = free_port(), free_port()
        while httpport == pgport:
            httpport = free_port()
        base_url = f"http://127.0.0.1:{httpport}/fhir"
        status.update({"base_url": base_url, "status": "starting_server"})
        write_json(output / "status.json", status)
        with (output / "postgres-init.log").open("w") as log:
            subprocess.run(["initdb", "-D", str(pgdata), "-U", "beyond_fhir", "-A", "trust", "--no-locale"],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        pg_started = True  # Also clean up if pg_ctl times out during startup.
        subprocess.run(["pg_ctl", "-D", str(pgdata), "-l", str(output / "postgres.log"), "-o",
                        f"-h 127.0.0.1 -p {pgport} -k ''", "-w", "-t", "60", "start"], check=True)
        subprocess.run(["createdb", "-h", "127.0.0.1", "-p", str(pgport), "-U", "beyond_fhir", "hapi"], check=True)
        env = dict(os.environ, BEYOND_FHIR_JDBC_URL=f"jdbc:postgresql://127.0.0.1:{pgport}/hapi",
                   BEYOND_FHIR_BASE_URL=base_url)
        # Let the verified manifest select the matching loader and nested libraries.
        processors = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
        command = ["java", "-Xms1g", "-Xmx10g", f"-XX:ActiveProcessorCount={processors}", "-Duser.timezone=UTC",
                   "-jar", str(staged_war), f"--spring.config.additional-location=file:{ROOT / 'configs' / 'hapi.yaml'}",
                   "--server.address=127.0.0.1", f"--server.port={httpport}"]
        with (output / "hapi.log").open("w") as log:
            process = subprocess.Popen(command, cwd=temporary, env=env, stdout=log, stderr=subprocess.STDOUT)
        client = FHIR(base_url, timeout=15)
        capability = wait_ready(client, process)
        write_json(output / "capability.json", capability)
        client.timeout = 180
        preflight_patient = paging_preflight_patient(snapshot, rows)
        if preflight_patient:
            status.update({"status": "paging_preflight", "preflight_patient": preflight_patient})
            write_json(output / "status.json", status)
            preflight_output = output / "preflight"
            preflight_output.mkdir()
            preflight_scope = scope_for_patients(snapshot, [preflight_patient])
            print(f"Checking large-search pagination first: {preflight_patient}", flush=True)
            import_snapshot(preflight_scope, client, preflight_output / "import.json", args.batch_size,
                            probe_search=True)
            preflight_rows = [row for row in rows if row["patient_id"] == preflight_patient]
            preflight_report = verify_dataset(preflight_scope, preflight_rows, client)
            write_json(preflight_output / "verification.json", preflight_report)
            if not preflight_report["valid"]:
                raise ValueError("Large-search pagination preflight failed; full import was not started. See preflight/verification.json")
            print(f"Pagination preflight passed: {preflight_report['questions_checked']} questions", flush=True)
        status["status"] = "importing"
        write_json(output / "status.json", status)
        import_snapshot(scoped, client, output / "import.json", args.batch_size, probe_search=True)
        status["status"] = "verifying"
        write_json(output / "status.json", status)
        print(f"Recomputing {len(rows)} questions against the live server", flush=True)
        report = verify_dataset(scoped, rows, client)
        write_json(output / "verification.json", report)
        if not report["valid"]:
            raise ValueError("Live FHIR verification failed; see verification.json")
        if after_verify is not None:
            status.update({"status": "running_pilot", "live_verification_passed": True})
            write_json(output / "status.json", status)
            print("Selected charts verified; starting zero-shot agent pilot", flush=True)
            after_verify(client, rows, snapshot)
        status.update({"status": "passed", "passed": True,
                       "questions_checked": report["questions_checked"],
                       "patients_checked": report["patients_checked"]})
        print(f"PASS: {report['questions_checked']} questions, {report['patients_checked']} patients", flush=True)
    except Exception as exc:
        status.update({"status": "failed", "passed": False, "error": str(exc)})
        # Preserve relevant request/server messages before shutdown fills the tail.
        hapi_log = output / "hapi.log"
        if hapi_log.exists():
            with hapi_log.open("rb") as handle:
                handle.seek(max(0, hapi_log.stat().st_size - 8000))
                print("HAPI output before cleanup:\n" + handle.read().decode("utf-8", errors="replace"), flush=True)
        error = exc
    finally:
        cleanup_errors = []
        try:
            stop_java(process)
        except (OSError, subprocess.SubprocessError) as exc:
            cleanup_errors.append(f"HAPI cleanup: {exc}")
        if pg_started:
            try:
                result = subprocess.run(["pg_ctl", "-D", str(temporary / "pgdata"), "-m", "fast",
                                         "-w", "-t", "30", "stop"],
                                        capture_output=True, text=True, timeout=45)
                status["postgres_stopped"] = result.returncode == 0
                if result.returncode:
                    cleanup_errors.append("PostgreSQL cleanup: " + result.stdout + result.stderr)
            except (OSError, subprocess.SubprocessError) as exc:
                cleanup_errors.append(f"PostgreSQL cleanup: {exc}")
        if temporary is not None and (not pg_started or status.get("postgres_stopped")):
            try:
                shutil.rmtree(temporary)
            except OSError as exc:
                cleanup_errors.append(f"Temporary directory cleanup: {exc}")
        try:
            sources.close()
        except Exception as exc:
            cleanup_errors.append(f"Source index cleanup: {exc}")
        if cleanup_errors:
            status.update({"cleanup_errors": cleanup_errors, "temporary_directory": str(temporary),
                           "status": "failed", "passed": False})
            error = error or RuntimeError("Server cleanup failed; see status.json")
        status["finished_at"] = now()
        write_json(output / "status.json", status)
        print(f"Live verification finished: {status['status']}", flush=True)
    if error is not None:
        raise error
    return status


@contextmanager
def verification_output(output=None):
    """Keep job files in temporary storage unless an output path is requested."""
    if output is not None:
        yield output
        return
    with tempfile.TemporaryDirectory(prefix="beyond-fhir-live-", dir=os.environ.get("SLURM_TMPDIR") or None) as temporary:
        path = Path(temporary) / "run"
        try:
            yield path
        finally:
            # Keep the useful outcome in stdout/Slurm logs, not a directory of exports.
            for name in ("status.json", "verification.json", "preflight/verification.json"):
                report = path / name
                if report.exists():
                    result = json.loads(report.read_text())
                    summary = {key: result[key] for key in ("status", "passed", "valid", "patients_checked", "questions_checked", "error", "cleanup_errors", "temporary_directory") if key in result}
                    if result.get("errors"):
                        summary["errors"] = result["errors"][:5]
                    print(f"{name}: {json.dumps(summary)}", flush=True)
            status_path = path / "status.json"
            if status_path.exists() and not json.loads(status_path.read_text()).get("passed"):
                for name in ("hapi.log", "postgres-init.log", "postgres.log"):
                    log = path / name
                    if log.exists():
                        with log.open("rb") as handle:
                            handle.seek(max(0, log.stat().st_size - 4000))
                            tail = handle.read().decode("utf-8", errors="replace")
                        print(f"Last {name} output:\n{tail}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    add_source_arguments(parser)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/cohort-200")
    parser.add_argument("--output", type=Path, help="Explicitly retain job artifacts here; otherwise temporary files are removed")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare()
        else:
            with verification_output(args.output) as output:
                args.output = output
                with interrupt_cleanup():
                    run_job(args)
    except (OSError, ValueError, KeyError, sqlite3.Error, subprocess.SubprocessError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
