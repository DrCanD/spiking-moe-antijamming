"""Recompute the measured KV260 contrasts from the archived raw power samples.

Requires Python 3.10+ and only the standard library. The original audit code is
run unchanged inside a temporary directory with its original relative layout.
"""

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def safe_extract(archive: Path, destination: Path):
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def main():
    with tempfile.TemporaryDirectory(prefix="kv260_raw_audit_") as directory:
        temp = Path(directory)
        audit = temp / "kv260_matched_review_20260929"
        audit.mkdir()
        shutil.copyfile(HERE / "audit_matched_results.py", audit / "audit_matched_results.py")
        for name, subdir in (("KV260_MATCHED_RESULTS.zip", "matched"), ("KV260_GATE_RESULTS.zip", "gate")):
            archive = ROOT / "evidence" / "hardware" / name
            target = audit / subdir
            target.mkdir()
            safe_extract(archive, target)
            original = temp / "upload" / name
            original.parent.mkdir(exist_ok=True)
            shutil.copyfile(archive, original)
        activity = temp / "matched_preparation_20260928" / "exp9b_activity_for_energy.json"
        activity.parent.mkdir()
        shutil.copyfile(HERE / "exp9b_activity_for_energy.json", activity)
        (audit / "audited").mkdir()
        subprocess.run([sys.executable, str(audit / "audit_matched_results.py")], check=True)
        outputs = ("independent_raw_audit.json", "measured_joint_energy.csv")
        for name in outputs:
            actual = (audit / "audited" / name).read_bytes()
            expected = (HERE / "reference" / name).read_bytes()
            if actual != expected:
                raise AssertionError(f"Recomputed {name} does not match the recorded audit")
            print(name, hashlib.sha256(actual).hexdigest())
        data = json.loads((audit / "audited" / outputs[0]).read_text())
        matched = data["campaigns"]["matched"]
        print("PASS: raw power means, paired intervals and archived outputs")
        print("Matched replay-rate range (MS/s):", matched["run_rate_range_msps"])


if __name__ == "__main__":
    main()
