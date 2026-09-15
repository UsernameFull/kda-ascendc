"""Every RTC compile of a kernels/v1 source carries the geometry defines.

A kernel that reads ``KDA_CHUNK`` but is compiled from a bare source string
does not fail to build: it keeps its ``#ifndef KDA_CHUNK 16`` default and
answers a C=32/64 host call with 16 rows per chunk.  That is a silent
correctness bug that looks like a speedup, and it has already happened once
(the "1.45 ms" C=64 probe that was really the C=16 kernel), so the invariant is
mechanical here: one compile helper, and this test fails if a call site
bypasses it.

Runs on the host: no NPU device and no launcher extension are needed.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "python" / "kda_ascendc_v1"
API = PKG / "api.py"
SCANNED_DIRS = ("python/kda_ascendc_v1", "tests", "tools", "benchmarks")
# aclab/ is a separate lab with its own launcher and its own kernels; it is
# not part of the v1 pipeline.
if str(ROOT / "python") not in sys.path:  # host-only: no NPU needed here
    sys.path.insert(0, str(ROOT / "python"))


def _iter_py_files():
    for rel in SCANNED_DIRS:
        for path in sorted((ROOT / rel).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def _bare_rtc_calls(path: Path):
    """(line, source) of every direct ``rtc_compile(...)`` call in ``path``."""
    # utf-8-sig: a few of the S12-S15 scripts still carry a BOM, which
    # ast.parse (unlike the import machinery) refuses.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "rtc_compile"):
            out.append((node.lineno, ast.unparse(node)))
    return out


def _functions(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.name, node.lineno, node.end_lineno))
    return spans


def test_bare_rtc_compile_only_lives_inside_the_helper():
    """``api._rtc`` is the one place a source string is handed to the compiler."""
    api_calls = _bare_rtc_calls(API)
    assert len(api_calls) == 1, [
        "api.py has %d direct rtc_compile calls; they must all go through _rtc"
        % len(api_calls)]
    line = api_calls[0][0]
    owner = [name for name, lo, hi in _functions(API) if lo <= line <= hi]
    assert owner == ["_rtc"], owner
    for path in _iter_py_files():
        if path == API:
            continue
        calls = _bare_rtc_calls(path)
        assert not calls, (
            "%s:%d compiles a kernel without the geometry defines (api._defines); "
            "use kda_ascendc_v1.api._rtc instead: %s"
            % (path.relative_to(ROOT), calls[0][0], calls[0][1]))


def _kernel_sources_without_comments(text: str) -> str:
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def test_every_chunk_dependent_kernel_is_reachable_from_the_api():
    """A kernels/v1 source that reads KDA_CHUNK must be compiled by api.py."""
    api_src = API.read_text(encoding="utf-8")
    referenced = set(re.findall(r"kernels/v1/([\w.]+\.cpp)", api_src))
    assert referenced, "api.py no longer names any kernel source"
    chunk_dependent = set()
    for src in sorted((ROOT / "kernels/v1").glob("*.cpp")):
        body = _kernel_sources_without_comments(src.read_text(encoding="utf-8"))
        if "KDA_CHUNK" in body:
            chunk_dependent.add(src.name)
    assert chunk_dependent, "no kernel reads KDA_CHUNK any more"
    missing = sorted(chunk_dependent - referenced)
    assert not missing, (
        "these kernels read KDA_CHUNK but api.py never compiles them, so nothing "
        "guarantees they get the defines prefix: %s" % missing)


def test_defines_and_compile_config_agree():
    """``_defines()`` is generated from the same numbers the profile reports."""
    from kda_ascendc_v1.api import (CHUNK, PERSIST_MAXH, _defines,
                                    compile_config, get_last_profile)

    defines = _defines()
    config = compile_config()
    for name in ("KDA_CHUNK", "KDA_MAXH", "KDA_SOLVE_WIDE_NCHUNK",
                 "KDA_SOLVE_WIDE_SUBB", "KDA_ASM_NCHUNK", "KDA_WU_NCHUNK"):
        assert name in defines, (name, defines)
        assert "#define %s %d" % (name, config[name]) in defines, (name, defines)
    assert config["KDA_CHUNK"] == CHUNK
    assert config["KDA_MAXH"] == PERSIST_MAXH
    profile = get_last_profile()
    assert profile["compile"] == config
