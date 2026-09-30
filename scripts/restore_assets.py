"""Restore and verify binary assets without depending on editable documentation."""

import hashlib
import json
from pathlib import Path
import zipfile

from update_manifests import safe_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "assets/manifest.json").read_text(encoding="utf-8"))
    archive_path = ROOT / safe_path(manifest["archive"])
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Checksum mismatch: binary asset archive")
    with zipfile.ZipFile(archive_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        names = [m.filename for m in members]
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
            raise ValueError("Archive membership differs from the asset manifest")
        for member in members:
            relative = safe_path(member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink: {member.filename}")
            content = archive.read(member)
            if hashlib.sha256(content).hexdigest() != manifest["files"][member.filename]:
                raise ValueError(f"Checksum mismatch: {member.filename}")
            destination = ROOT / relative
            if not destination.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError(f"Asset destination escapes repository: {member.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    print(f"PASS: restored and verified {len(members)} binary assets")


if __name__ == "__main__":
    main()
