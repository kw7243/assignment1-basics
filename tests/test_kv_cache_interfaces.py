"""Check standalone constructor compatibility and cached-call requirements."""

import pytest
import torch


@pytest.fixture(autouse=True)
def interface_inference_mode():
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(42)
        yield


@pytest.fixture(params=["attention", "block"])
def component_factory(request):
    from cs336_basics.attention import MultiHeadSelfAttention
    from cs336_basics.transformer import TransformerBlock

    def create(**kwargs):
        # Preserve the existing positional order of pos_layer, device, dtype.
        if request.param == "attention":
            return MultiHeadSelfAttention(8, 2, None, torch.device("cpu"), torch.float64, **kwargs).eval()
        return TransformerBlock(8, 2, 12, None, torch.device("cpu"), torch.float64, **kwargs).eval()

    return create


def test_kv_cache_interface_legacy_uncached_constructor(component_factory):
    standalone = component_factory()
    indexed = component_factory(layer_idx=1)
    indexed.load_state_dict(standalone.state_dict())
    inputs = torch.randn(2, 3, 8, dtype=torch.float64)

    actual = standalone(inputs.clone())
    expected = indexed(inputs.clone())

    assert actual.dtype == torch.float64
    assert actual.device == torch.device("cpu")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_kv_cache_interface_cached_call_requires_index(component_factory):
    from cs336_basics.kv_cache import KVCache

    component = component_factory()
    cache = KVCache(
        num_layers=3,
        batch_size=2,
        num_kv_heads=2,
        max_seq_len=8,
        d_head=4,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="layer_idx"):
        component(torch.zeros(2, 3, 8, dtype=torch.float64), kv_cache=cache)

    assert [cache.get_length(layer) for layer in range(3)] == [0, 0, 0]
