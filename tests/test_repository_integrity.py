"""Regression checks for portable asset restoration and manifest maintenance."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class RepositoryIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        for name in ("restore_assets.py", "update_manifests.py"):
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        ignore = (ROOT / ".gitignore").read_text().split("# BEGIN restored assets", 1)[0]
        (self.root / ".gitignore").write_text(ignore)
        (self.root / "README.md").write_text("Test distribution\n")
        assets = self.root / "assets"
        assets.mkdir()
        self.contents = {"hardware/kv260/vectors/input.bin": b"input samples",
                         "hardware/kv260/vectors/tracked.bin": b"tracked samples",
                         "simulation/frame_receiver/models/model.bin": b"model parameters"}
        with zipfile.ZipFile(assets / "binary-evidence.zip", "w") as archive:
            for name, content in self.contents.items():
                archive.writestr(name, content)
        tracked = self.root / "hardware/kv260/vectors/tracked.bin"
        tracked.parent.mkdir(parents=True)
        tracked.write_bytes(self.contents["hardware/kv260/vectors/tracked.bin"])
        self.git("init", "-q")
        self.git("add", "-A")
        self.run_script("update_manifests.py")
        self.git("add", "-A")
        self.run_script("update_manifests.py", "--check")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True)

    def run_script(self, script, *args, succeeds=True):
        result = subprocess.run([sys.executable, "scripts/" + script, *args],
                                cwd=self.root, text=True, capture_output=True)
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_readme_edit_does_not_block_restoration(self):
        (self.root / "README.md").write_text("Edited documentation\n")
        self.run_script("restore_assets.py")
        for name, content in self.contents.items():
            self.assertEqual((self.root / name).read_bytes(), content)
        result = self.run_script("update_manifests.py", "--check", succeeds=False)
        self.assertIn("MANIFEST_SHA256.txt", result.stdout + result.stderr)
        self.run_script("update_manifests.py")
        self.run_script("update_manifests.py", "--check")

    def test_restoration_and_new_receipts_leave_no_untracked_files(self):
        self.run_script("restore_assets.py")
        receipt = self.root / "hardware/kv260/evidence/matched_regeneration.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text('{"receipt_kind":"post_campaign_regeneration"}\n')
        self.assertEqual(self.git("ls-files", "--others", "--exclude-standard"), "")
        self.assertEqual(self.git("diff", "--name-only"), "")
        self.run_script("update_manifests.py", "--check")

    def test_damaged_archive_is_rejected(self):
        archive = self.root / "assets/binary-evidence.zip"
        with archive.open("ab") as handle:
            handle.write(b"unrecorded bytes")
        result = self.run_script("restore_assets.py", succeeds=False)
        self.assertIn("Checksum mismatch", result.stderr)

    def test_wrong_member_checksum_is_rejected(self):
        path = self.root / "assets/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"]["hardware/kv260/vectors/input.bin"] = "0" * 64
        path.write_text(json.dumps(manifest))
        result = self.run_script("restore_assets.py", succeeds=False)
        self.assertIn("Checksum mismatch: hardware/kv260/vectors/input.bin", result.stderr)

    def test_source_release_checks_without_git_metadata(self):
        shutil.rmtree(self.root / ".git")
        self.run_script("update_manifests.py", "--check")
        self.run_script("restore_assets.py")


if __name__ == "__main__":
    unittest.main()
