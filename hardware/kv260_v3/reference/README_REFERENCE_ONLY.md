# Historical snapshot: do not execute these scripts

This directory contains the previous project sources retrieved for provenance.
Use ../build_kv260.py and ../board/run_gate.py for the new gate experiment.

The previous measure.py divides mW/MSps by another 1000 when labelling nJ/sample.
That is a unit error. The new runner uses mW/MSps directly, and separates total
SOM input, idle-adjusted, and D0-subtracted operational estimates. Previous
arithmetic pacing also consumed switching power; the new top uses explicit HLS
waits. These changes mean old and new absolute numbers must not simply be pooled.

README_old.md describes the old package, not validation status of this new one.
The frozen coefficients in moe_params.h are used by fixed_reference.py.
