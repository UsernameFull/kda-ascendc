#!/usr/bin/env bash
# The production performance gate: [1,8192,96,128], persistent_loop.
#
# The golden (benchmarks/golden/) fixes the geometry; --kda-chunk states the
# build this script means, and --gate refuses a mismatch before spending any
# device time.  KDA_CHUNK=... overrides it.  A recording needs --kda-chunk 64
# explicitly (the benchmark refuses to guess): the C=16 build answers the same
# call with 11.5 ms and four times K2's step count.
#
#   bash tools/run_bench_gate.sh                      # check (exit 1 on a regression)
#   bash tools/run_bench_gate.sh --update-golden      # re-record after an intended change
set -u
cd "$(dirname "$0")/.."
exec python3 benchmarks/bench_fla_compare.py --shape 1,8192,96,128 \
    --impl "${KDA_GATE_IMPL:-fla-alog,ascendc}" --ascendc-modes persistent_loop \
    --chunk-sizes 64 --kda-chunk "${KDA_CHUNK:-64}" --gate "$@"
