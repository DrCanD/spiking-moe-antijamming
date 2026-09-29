"""Check delivered file integrity; generated build/result files are not part of this list."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PACKAGE_SHA256.json').read_text())
failed=[]
for rel,wanted in manifest.items():
    p=root/rel
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=wanted:failed.append(rel)
if failed:raise SystemExit('Missing/changed delivered files: '+', '.join(failed))
print('PASS: delivered files',len(manifest))
