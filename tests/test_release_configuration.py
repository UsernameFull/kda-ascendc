import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")

import kda_ascendc_v1.api as api


pytestmark = pytest.mark.npu


def test_compile_geometry_is_self_consistent():
    assert api.CHUNK in api.SUPPORTED_CHUNKS
    assert api.SOLVE_WIDE_NCH == api.SOLVE_WIDE_NCHUNK // api.SOLVE_WIDE_SUBB
    config = api.compile_config()
    assert config["KDA_CHUNK"] == api.CHUNK
    assert config["KDA_SOLVE_WIDE_NCHUNK"] == api.SOLVE_WIDE_NCHUNK


def test_solve_padding_covers_whole_wiring_units():
    unit = api.math.lcm(api.SOLVE_WIDE_NCH, api.ASM_NCHUNK, api.WU_NCHUNK)
    for chunks in range(1, 5 * unit + 1):
        padded = api._solve_padded_chunks(chunks)
        assert padded >= chunks
        assert padded % unit == 0


def test_kernel_defines_are_guarded_against_runtime_config():
    for source_path, _ in api._SOURCES:
        text = (api.ROOT / source_path).read_text(encoding="utf-8-sig")
        assert text.count("#ifndef KDA_CHUNK") == 1
        assert text.count("#endif") >= 1


def test_public_entry_rejects_non_npu_inputs_before_compile():
    steps = 2 * api.CHUNK
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    q = torch.zeros(1, steps, 1, api.D, dtype=torch.bfloat16)
    k = q.clone()
    v = q.clone()
    g = torch.zeros(1, steps, 1, api.D, dtype=torch.float32)
    beta = torch.zeros(1, steps, 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="requires NPU"):
        api.kda_bt16_fwd_ascendc(q, k, v, g, beta)

    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    g = g.to(device)
    beta = beta.to(device)
    with pytest.raises(ValueError, match="unsupported k2_mode"):
        api.kda_bt16_fwd_ascendc(
            q, k, v, g, beta,
            k2_mode="not_persistent_loop",
        )
