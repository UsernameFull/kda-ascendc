#!/bin/bash
# One-time shims that make the CANN ascendc cmake machinery + mskl launcher work
# on this box (read-only /usr/local/Ascend, nonstandard toolchain layout).
set -e

A=/usr/local/Ascend/cann-9.1.0
DK=$A/aarch64-linux/ascendc_devkit

# 1. device compiler symlinks expected by bisheng_config.cmake
mkdir -p $DK/ccec_compiler/bin
for t in bisheng bishengir-compile ld.lld llvm-ar llvm-strip llvm-objcopy llvm-link ccec; do
  ln -sf $A/bin/$t $DK/ccec_compiler/bin/$t
done

# 2. include trees expected by the generated compile lines
ln -sfn $A/aarch64-linux/asc $DK/asc
T=$(find $A -maxdepth 4 -type d -name tikcfw | head -1)
mkdir -p $DK/tikcpp && ln -sfn $T $DK/tikcpp/tikcfw

# 3. writable CANN root for mskl (FileChecker requires W_OK)
SH=$(dirname "$0")/cannshim
mkdir -p $SH
for d in lib64 include bin compiler tikcpp asc ascendc python devlib fwkacllib; do
  ln -sfn $A/aarch64-linux/$d $SH/$d
done

echo "shims ready: build with cmake -B build -S . ; run with ASCEND_HOME_PATH=$SH"
