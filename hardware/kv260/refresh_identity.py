"""For source developers: invalidate any build receipt after editing and revalidate."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    h=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((ROOT/'hls/src').glob('*')) if p.name!='build_id.h')).hexdigest()
    import re
    proto=int(re.search(r'#define GATE_PROTOCOL (0x[0-9a-fA-F]+)u',(ROOT/'hls/src/gate_lut.h').read_text()).group(1),16)
    magic=int(re.search(r'#define RES_MAGIC (0x[0-9a-fA-F]+)u',(ROOT/'hls/src/moe_types.h').read_text()).group(1),16)
    identity=dict(build_id=int(h[:8],16),source_digest=h,protocol=proto,magic=magic,default_replicas=4,version='v3')
    (ROOT/'build_identity.json').write_text(json.dumps(identity,indent=2))
    (ROOT/'hls/src/build_id.h').write_text('#ifndef GATE_BUILD_ID_H\n#define GATE_BUILD_ID_H\n#define GATE_BUILD_ID 0x'+h[:8]+'u\n#endif\n')
    (ROOT/'board/build_receipt.json').unlink(missing_ok=True)
    print('Identity refreshed; rerun native/ap_int/RTL/board verification before measuring.',identity)
if __name__=='__main__':main()
