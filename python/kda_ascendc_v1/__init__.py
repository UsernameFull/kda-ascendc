"""AscendC (RTC-compiled) KDA v1 backend.

The public entry point serves one K2 implementation: ``persistent_loop``, the
chunk-generic device-side loop.  ``kda_ascendc_v1.experimental`` carries the
C=16-only historical kernels for benchmarking.
"""
from .api import CHUNK, PERSISTENT_LOOP, get_last_profile, kda_bt16_fwd_ascendc
from .layout import pack_tokens, unpack_tokens

__all__ = [
    "CHUNK",
    "PERSISTENT_LOOP",
    "get_last_profile",
    "kda_bt16_fwd_ascendc",
    "pack_tokens",
    "unpack_tokens",
]
__version__ = "0.1.0"
