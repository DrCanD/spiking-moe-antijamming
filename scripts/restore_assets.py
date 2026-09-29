"""Restore archived binary evidence to its original, checksum-verified paths."""

from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "assets" / "binary-evidence.zip"
MANIFEST = ROOT / "MANIFEST_SHA256.txt"


def main():
    with zipfile.ZipFile(ARCHIVE) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise ValueError(f"Damaged archive member: {bad}")
        for member in bundle.infolist():
            relative = Path(member.filename)
            if member.is_dir():
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            destination = ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)

    checked = 0
    for line in MANIFEST.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
        checked += 1
    print(f"PASS: restored binary evidence and verified {checked} files")


if __name__ == "__main__":
    main()
