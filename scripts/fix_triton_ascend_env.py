#!/usr/bin/env python3
"""Repair the local triton-ascend install so the KDA Triton kernels can run.

The Ascend build of Triton breaks in two independent, recurring ways on this
machine:

1. **Clobbered install.**  Installing or upgrading ``torch`` pulls the vanilla
   ``triton`` wheel, which overwrites ``triton/_C/libtriton.so`` and the Python
   sources of triton-ascend.  ``import triton`` then fails while discovering
   backends with ``ImportError: cannot import name 'ascend' from
   'triton._C.libtriton'``.  ``--wheel`` reinstalls the Ascend wheel after
   moving the clobbered tree aside.

2. **CANN >= 9.1 enumerator rename.**  ``triton/backends/ascend/npu_utils.cpp``
   maps ``"WARP_STACK_SIZE"`` to
   ``rtLimitType_t::RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE``.  CANN 9.1 dropped
   that enumerator (it ships ``RT_LIMIT_TYPE_SIMT_STACK_SIZE`` instead), so the
   helper extension that the driver builds at first use fails with
   ``error: 'RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE' is not a member of
   'rtLimitType_t'``.  A guarded ``#define`` is inserted after the last
   ``#include``; it is inert on CANN releases that still define the old name
   and only affects the public ``set_device_limit`` helper, which is not used
   by the KDA kernels.

Everything the script moves or rewrites is kept, so the change is reversible.

Usage::

    python scripts/fix_triton_ascend_env.py --check    # diagnose only
    python scripts/fix_triton_ascend_env.py            # repair + verify
    python scripts/fix_triton_ascend_env.py --wheel /path/to/triton_ascend-*.whl
"""

from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

SHIM_MARKER = "KDA_TRITON_ASCEND_CANN_COMPAT"
SHIM_BLOCK = (
    f"// {SHIM_MARKER}: CANN >= 9.1 renamed RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE.\n"
    "#ifndef RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE\n"
    "#ifdef RT_LIMIT_TYPE_SIMT_STACK_SIZE\n"
    "#define RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE RT_LIMIT_TYPE_SIMT_STACK_SIZE\n"
    "#else\n"
    "#define RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE RT_LIMIT_TYPE_STACK_SIZE\n"
    "#endif\n"
    "#endif\n"
)
VENDOR_WHEEL_GLOBS = (
    "/data/*/triton-ascend-vendor/triton_ascend-*.whl",
    "/data/triton-ascend-vendor/triton_ascend-*.whl",
)
MIRROR = "https://repo.huaweicloud.com/repository/pypi/simple"


def site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def triton_package_dir() -> Path | None:
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("triton")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def ascend_backend_available() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-c", "from triton._C.libtriton import ascend"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True, "triton._C.libtriton.ascend imports"
    detail = (proc.stderr or proc.stdout).strip().splitlines()
    return False, detail[-1] if detail else f"exit code {proc.returncode}"


def find_wheel(explicit: str | None, allow_download: bool) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"wheel not found: {path}")
        return path
    env = os.environ.get("TRITON_ASCEND_WHEEL")
    if env:
        path = Path(env)
        if not path.is_file():
            raise SystemExit(f"$TRITON_ASCEND_WHEEL is not a file: {path}")
        return path
    for pattern in VENDOR_WHEEL_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[-1])
    if not allow_download:
        return None
    dest = Path("/tmp/triton_ascend_wheel")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[repair] downloading triton-ascend from {MIRROR}")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "download", "triton-ascend", "--no-deps",
         "-d", str(dest), "-i", MIRROR],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return None
    hits = sorted(dest.glob("triton_ascend-*.whl"))
    return hits[-1] if hits else None


def stale_install_paths() -> list[Path]:
    root = site_packages()
    paths = [root / "triton"]
    paths += sorted(root.glob("triton-*.dist-info"))
    paths += sorted(root.glob("triton_ascend-*.dist-info"))
    return [p for p in paths if p.exists()]


def repair_install(wheel: Path, dry_run: bool) -> None:
    backup = Path("/tmp") / f"triton_ascend_repair_{time.strftime('%Y%m%d_%H%M%S')}"
    targets = stale_install_paths()
    print(f"[repair] wheel   : {wheel}")
    print(f"[repair] backup  : {backup}")
    for target in targets:
        print(f"[repair] move    : {target}")
    if dry_run:
        return
    backup.mkdir(parents=True, exist_ok=True)
    for target in targets:
        shutil.move(str(target), str(backup / target.name))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
    )


def patch_npu_utils(triton_dir: Path, dry_run: bool) -> str:
    path = triton_dir / "backends" / "ascend" / "npu_utils.cpp"
    if not path.is_file():
        return f"skipped, {path} not found"
    text = path.read_text(encoding="utf-8")
    if SHIM_MARKER in text:
        return f"already patched: {path}"
    if "RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE" not in text:
        return f"skipped, no reference to patch in {path}"
    lines = text.splitlines(keepends=True)
    last_include = max(i for i, line in enumerate(lines) if line.lstrip().startswith("#include"))
    lines.insert(last_include + 1, SHIM_BLOCK)
    if dry_run:
        return f"would patch: {path}"
    backup = path.with_name(path.name + f".bak-{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(path, backup)
    path.write_text("".join(lines), encoding="utf-8")
    return f"patched: {path} (backup {backup.name})"


def verify() -> tuple[bool, str]:
    code = (
        "import triton\n"
        "from triton.backends.ascend.driver import NPUUtils\n"
        "utils = NPUUtils()\n"
        "print('driver helper:', utils.npu_utils_mod.__file__)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only diagnose, change nothing")
    parser.add_argument("--dry-run", action="store_true", help="print the planned actions")
    parser.add_argument("--wheel", help="triton-ascend wheel to reinstall from")
    parser.add_argument("--no-download", action="store_true", help="never fetch the wheel from the mirror")
    parser.add_argument("--no-verify", action="store_true", help="skip building the driver helper")
    args = parser.parse_args()

    ok, detail = ascend_backend_available()
    triton_dir = triton_package_dir()
    print(f"[check] triton package  : {triton_dir}")
    print(f"[check] ascend backend  : {'ok' if ok else 'BROKEN'} ({detail})")

    if args.check:
        if not ok:
            wheel = find_wheel(args.wheel, allow_download=False)
            print(f"[check] repair wheel    : {wheel or 'not found (pass --wheel)'}")
        return 0 if ok else 1

    if not ok or args.wheel:
        wheel = find_wheel(args.wheel, allow_download=not args.no_download)
        if wheel is None:
            print("[repair] no triton-ascend wheel available; pass --wheel or allow the mirror download")
            return 1
        repair_install(wheel, args.dry_run)
        triton_dir = triton_package_dir()

    if triton_dir is None:
        print("[repair] triton package still missing after reinstall")
        return 1
    print(f"[patch ] {patch_npu_utils(triton_dir, args.dry_run)}")

    if args.dry_run or args.no_verify:
        return 0
    ok, detail = verify()
    print(f"[verify] driver helper  : {'ok' if ok else 'FAILED'}")
    if not ok:
        print(detail)
        return 1
    print(detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
