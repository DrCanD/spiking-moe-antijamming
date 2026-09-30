"""Restore the 44 tail-input word blocks from the archived native test file."""
from pathlib import Path
import argparse
import json
import struct

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'hls/tb/rf_tests.bin')
    ap.add_argument('--output',type=Path,default=ROOT/'vectors')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with a.input.open('rb') as f:
        magic,count=struct.unpack('<II',f.read(8))
        if magic!=0x52465453:raise ValueError('Invalid test-vector header')
        for _ in range(count):
            name=f.read(32).split(b'\0')[0].decode();n,=struct.unpack('<I',f.read(4))
            w=struct.unpack('<1024I',f.read(4096));f.read(72)
            fixed=struct.unpack('<20I',f.read(80));random=struct.unpack('<20I',f.read(80))
            stats=dict(n=n,count=w[0],isi_n=w[1],isi_sum=w[2],isi_sq_sum=w[3]+(w[4]<<32),
                       sf_cnt=list(w[5:25]),sf_isi_sum=list(w[25:45]),sf_isi_cnt=list(w[45:65]))
            row=dict(spike_stats=stats,rf_counts=[list(w[65+16*i:65+16*(i+1)]) for i in range(20)])
            if name.startswith(('fixed_','random_')):
                protocol=name.split('_')[0];row['protocol']=protocol
                row['router_verdict']=['broadband','narrowband','none','pulse','sweep'][(fixed if protocol=='fixed' else random)[0]]
            (a.output/(name+'_expected.json')).write_text(json.dumps(row,indent=2)+'\n')
        if f.read(1):raise ValueError('Unexpected trailing test bytes')
    print('Restored classifier vectors:',count)


if __name__=='__main__':
    main()
