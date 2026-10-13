import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")

from kda_ascendc_v1.api import CHUNK, SUPPORTED_CHUNKS, get_last_profile, kda_bt16_fwd_ascendc
from flashkda_reference import flashkda_reference_forward


pytestmark = pytest.mark.npu

DIM = 128
HEADS = 2


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    error = float((actual.float() - reference.float()).abs().max())
    return error / (float(reference.float().abs().max()) + 1e-12)


def _assert_supported_build():
    if CHUNK not in SUPPORTED_CHUNKS:
        pytest.skip(f"KDA_CHUNK={CHUNK} is not a supported release build")


@pytest.mark.parametrize("chunks,with_state", [(2, False), (4, True)])
def test_release_matches_flashkda_reference(chunks, with_state):
    _assert_supported_build()
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    steps = chunks * CHUNK
    generator = torch.Generator().manual_seed(240920 + chunks)

    def cpu_randn(*shape, multiplier=1.0):
        return torch.randn(*shape, generator=generator) * multiplier

    q = cpu_randn(1, steps, HEADS, DIM, multiplier=0.2).to(device=device, dtype=torch.bfloat16)
    k = cpu_randn(1, steps, HEADS, DIM, multiplier=0.2).to(device=device, dtype=torch.bfloat16)
    v = cpu_randn(1, steps, HEADS, DIM, multiplier=0.1).to(device=device, dtype=torch.bfloat16)
    g = cpu_randn(1, steps, HEADS, DIM, multiplier=0.1).to(device=device, dtype=torch.float32)
    beta = cpu_randn(1, steps, HEADS).to(device=device, dtype=torch.float32)
    a_log = torch.linspace(-1.0, 0.2, HEADS, device=device, dtype=torch.float32)
    bias = cpu_randn(HEADS, DIM, multiplier=0.03).to(device=device, dtype=torch.float32)
    initial_state = None
    if with_state:
        initial_state = cpu_randn(1, HEADS, DIM, DIM, multiplier=0.01).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        output, state = kda_bt16_fwd_ascendc(
            q, k, v, g, beta,
            scale=DIM ** -0.5,
            initial_state=initial_state,
            output_final_state=True,
            A_log=a_log,
            bias=bias,
            lower_bound=-1.0,
        )
        torch.npu.synchronize()

    reference_output, reference_state = flashkda_reference_forward(
        q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(), DIM ** -0.5,
        a_log.cpu(), bias.cpu(), lower_bound=-1.0, chunk_size=CHUNK,
        initial_state=None if initial_state is None else initial_state.cpu(),
    )
    assert output.shape == (1, steps, HEADS, DIM)
    assert state.shape == (1, HEADS, DIM, DIM)
    assert torch.isfinite(output).all()
    assert torch.isfinite(state).all()
    assert _relative_error(output, reference_output) < 2e-2
    assert _relative_error(state, reference_state) < 2e-2


def test_release_is_deterministic_and_single_launch_k2():
    _assert_supported_build()
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    steps = 2 * CHUNK
    generator = torch.Generator().manual_seed(250920)

    def cpu_randn(*shape, multiplier=1.0):
        return torch.randn(*shape, generator=generator) * multiplier

    q = cpu_randn(1, steps, HEADS, DIM, multiplier=0.2).to(device=device, dtype=torch.bfloat16)
    k = cpu_randn(1, steps, HEADS, DIM, multiplier=0.2).to(device=device, dtype=torch.bfloat16)
    v = cpu_randn(1, steps, HEADS, DIM, multiplier=0.1).to(device=device, dtype=torch.bfloat16)
    g = cpu_randn(1, steps, HEADS, DIM, multiplier=0.1).to(device=device, dtype=torch.float32)
    beta = cpu_randn(1, steps, HEADS).to(device=device, dtype=torch.float32)
    kwargs = dict(
        A_log=torch.zeros(HEADS, device=device, dtype=torch.float32),
        bias=torch.zeros(HEADS, DIM, device=device, dtype=torch.float32),
        lower_bound=-1.0,
        output_final_state=True,
    )
    with torch.inference_mode():
        first_output, first_state = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kwargs)
        second_output, second_state = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kwargs)
        torch.npu.synchronize()
    assert torch.equal(first_output, second_output)
    assert torch.equal(first_state, second_state)
    assert get_last_profile()["launch_counts"].get("kda_k2_persistent_loop") == 1
