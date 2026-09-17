#!/usr/bin/env bash
# The CHUNK x shape correctness matrix, one process per build.
#
# The chunk size is a compile-time constant of the RTC kernels, so a build is
# only ever checked against the shapes its CHUNK divides; this runs the same
# matrix at C=16, C=32 and C=64 and reports which builds pass.  C=32 is a
# known-broken build (see api.SUPPORTED_CHUNKS), so the host refuses it and its
# leg reports the refusal rather than a pass.
#
# tests/test_c64_gate_overflow.py rides along: it pins the C=64 band split of
# the gate reference (docs section 11.23) and its (1, 0) Gram block, so it has
# to pass at C=64 and at C=16 from the same source.
#
#   bash tools/run_chunk_matrix.sh            # matrix + stability gate
#   KDA_STRESS_ITERS=100 bash tools/run_chunk_matrix.sh
set -u
cd "$(dirname "$0")/.."
rc=0
for chunk in 16 32 64; do
    echo "=== KDA_CHUNK=${chunk} ==="
    if KDA_CHUNK="${chunk}" python3 -m pytest \
            tests/test_chunk_shape_matrix.py tests/test_stability_gate.py \
            tests/test_c64_gate_overflow.py -q; then
        if [ "${chunk}" = "32" ]; then
            echo "REFUSED KDA_CHUNK=32 (known-broken build: the api guard fires and the tests skip)"
        else
            echo "PASS KDA_CHUNK=${chunk}"
        fi
    else
        echo "FAIL KDA_CHUNK=${chunk}"
        rc=1
    fi
done
exit "${rc}"
