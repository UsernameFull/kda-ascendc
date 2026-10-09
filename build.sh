#!/usr/bin/env bash
# Build the KDA Ascend C release tree: launcher + RTC kernel precompile.
#
#   ./build.sh                  launcher, then precompile all five kernels
#   ./build.sh --launcher-only  only the launcher (no NPU required)
#   ./build.sh --kernels-only   only the kernel precompile (launcher exists)
#
# Environment: KDA_CHUNK (default 64), PY (default python3),
#   ASCEND_RT_VISIBLE_DEVICES - physical device for the precompile step,
#   which opens in-process device 0 (torch.npu.set_device(0)).
#
# The launcher lands in
# build/S02_clean/torch_extensions/kda_ascendc_v1_launcher, where
# python/kda_ascendc_v1/api.py imports it from.
set -euo pipefail

cd "$(dirname "$0")"

KDA_CHUNK="${KDA_CHUNK:-64}"
PY="${PY:-python3}"
LAUNCHER_DIR="build/S02_clean/torch_extensions/kda_ascendc_v1_launcher"
LAUNCHER_SO="$LAUNCHER_DIR/kda_ascendc_v1_launcher.so"

mode=all
for arg in "$@"; do
  case "$arg" in
    --all) mode=all ;;
    --launcher-only | --launcher) mode=launcher ;;
    --kernels-only | --kernels) mode=kernels ;;
    -h | --help) sed -n '2,14p' "$0"; exit 0 ;;
    *)
      echo "build.sh: unknown argument: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

if [[ "$mode" == all || "$mode" == launcher ]]; then
  echo "==> [launcher] -> $LAUNCHER_SO"
  LAUNCHER_NAME=kda_ascendc_v1_launcher bash aclab/launcher/build_launcher.sh
  [[ -f "$LAUNCHER_SO" ]] || { echo "build.sh: $LAUNCHER_SO was not produced" >&2; exit 1; }
fi

if [[ "$mode" == all || "$mode" == kernels ]]; then
  [[ -f "$LAUNCHER_SO" ]] || {
    echo "build.sh: launcher missing at $LAUNCHER_SO - run ./build.sh --launcher-only first" >&2
    exit 1
  }
  echo "==> [kernels] precompiling all five kernels at KDA_CHUNK=$KDA_CHUNK"
  KDA_CHUNK="$KDA_CHUNK" "$PY" tools/compile_all_server.py
fi

echo "build.sh: OK (KDA_CHUNK=$KDA_CHUNK)"
