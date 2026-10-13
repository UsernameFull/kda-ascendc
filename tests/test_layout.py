import pytest
import torch

from kda_ascendc_v1.layout import pack_tokens, unpack_tokens


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("batch,steps,heads,dim,chunk_size", [
    (1, 16, 1, 128, 16),
    (2, 64, 3, 128, 32),
    (1, 256, 4, 128, 64),
])
def test_pack_and_unpack_round_trip(batch, steps, heads, dim, chunk_size):
    original = torch.randn(batch, steps, heads, dim)
    packed = pack_tokens(original, chunk=chunk_size)
    restored = unpack_tokens(packed, batch, steps, heads, chunk=chunk_size)
    assert packed.shape == (batch * heads * steps // chunk_size, chunk_size, dim)
    assert torch.equal(original, restored)
