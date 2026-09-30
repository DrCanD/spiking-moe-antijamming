# Copy this entire file into one Colab cell. CPU runtime; no GPU is required.
MODE = "pilot"  # "verify", "pilot" (720 development streams), or "full" (+1200 validation)
WORKERS = 2
OUTPUT_ROOT = "/content/antijamming_runs"  # Set a persistent output directory for long runs.

from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import subprocess
import sys

REPOSITORY = "https://github.com/DrCanD/spiking-moe-antijamming.git"
SOURCE_COMMIT = "cbf39e8c5c9205bef08d4509f5899c10af0746c6"
assert MODE in ("verify", "pilot", "full"), "MODE must be verify, pilot, or full"
assert WORKERS >= 1

workspace = Path("/content") if Path("/content").is_dir() else Path.cwd()
repo = workspace / ("antijamming_source_" + SOURCE_COMMIT[:12])
env_dir = workspace / "antijamming_python"
run_dir = Path(OUTPUT_ROOT).expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
run_dir.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")


def run(args, cwd=None):
    print("RUN:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


if not (repo / ".git").exists():
    run(["git", "init", repo])
    run(["git", "-C", repo, "remote", "add", "origin", REPOSITORY])
    run(["git", "-C", repo, "fetch", "--depth", "1", "origin", SOURCE_COMMIT])
    run(["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"])
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
if head != SOURCE_COMMIT:
    raise RuntimeError("Source checkout differs from the frozen experiment commit")

run([sys.executable, repo / "scripts/restore_assets.py"])
python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not python.exists():
    run([sys.executable, "-m", "venv", "--without-pip", env_dir])
run([sys.executable, "-m", "pip", "--python", python, "install", "-r",
     repo / "simulation/streaming/frozen_v4/requirements.txt"])

source = repo / "simulation/streaming"
print("Output:", run_dir, "| CPU workers:", WORKERS, "| MODE:", MODE, flush=True)
run([python, source / "verify_v5.py", "--output", run_dir / "verification.json"])
if MODE in ("pilot", "full"):
    development = run_dir / "development"
    run([python, source / "run_v5.py", "--profile", "development", "--output", development,
         "--workers", WORKERS])
    if MODE == "full":
        run([python, source / "run_v5.py", "--profile", "validation", "--output", run_dir / "validation",
             "--selection", development / "selection.json", "--workers", WORKERS])

archive = shutil.make_archive(str(run_dir.parent / (run_dir.name + "_RESULTS_REVIEW")), "zip", run_dir)
print("COMPLETED:", archive, flush=True)
