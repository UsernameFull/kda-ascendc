import pytest
import torch

from flashkda_reference import flashkda_reference_forward


pytestmark = pytest.mark.unit


def _inputs(steps, heads, dim, chunk_size, seed):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape, multiplier=1.0):
        return torch.randn(*shape, generator=generator) * multiplier

    return {
        "q": randn(1, steps, heads, dim, multiplier=0.2).to(torch.bfloat16),
        "k": randn(1, steps, heads, dim, multiplier=0.2).to(torch.bfloat16),
        "v": randn(1, steps, heads, dim, multiplier=0.1).to(torch.bfloat16),
        "g": randn(1, steps, heads, dim, multiplier=0.1),
        "beta": randn(1, steps, heads),
        "scale": dim ** -0.5,
        "A_log": torch.linspace(-1.0, 0.2, heads),
        "dt_bias": randn(heads, dim, multiplier=0.03),
        "lower_bound": -1.0,
        "chunk_size": chunk_size,
    }


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_reference_matches_across_flashkda_chunk_sizes(chunk_size):
    inputs = _inputs(4 * chunk_size, 2, 64, chunk_size, seed=1000 + chunk_size)
    initial_state = torch.randn(1, 2, 64, 64) * 0.01
    output, state = flashkda_reference_forward(
        **inputs, initial_state=initial_state
    )
    assert output.shape == (1, 4 * chunk_size, 2, 64)
    assert state.shape == (1, 2, 64, 64)
    assert torch.isfinite(output).all()
    assert torch.isfinite(state).all()


def test_reference_applies_the_initial_state():
    inputs = _inputs(16, 1, 32, 16, seed=17)
    zero_state = torch.zeros(1, 1, 32, 32)
    non_zero_state = torch.randn(1, 1, 32, 32) * 0.1
    zero_output, zero_final = flashkda_reference_forward(
        **inputs, initial_state=zero_state
    )
    warm_output, warm_final = flashkda_reference_forward(
        **inputs, initial_state=non_zero_state
    )
    assert not torch.allclose(zero_output, warm_output)
    assert not torch.allclose(zero_final, warm_final)


def test_reference_validates_shapes():
    inputs = _inputs(32, 1, 32, 16, seed=23)
    bad_inputs = dict(inputs)
    bad_inputs["k"] = inputs["k"][:, :-1]
    with pytest.raises(ValueError, match="identical"):
        flashkda_reference_forward(**bad_inputs)
    bad_chunk = dict(inputs)
    bad_chunk["chunk_size"] = 7
    with pytest.raises(ValueError, match="divisible"):
        flashkda_reference_forward(**bad_chunk)
