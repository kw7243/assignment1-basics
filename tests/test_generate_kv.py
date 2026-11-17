"""Test cached generation separately from the existing naive decoder tests."""

import pytest
import torch
import torch.nn as nn

from .adapters import run_decode_step, run_generate_kv, run_generate_kv_with_model, run_generate_naive, run_prefill
from .kv_cache_integration_adapters import get_transformer_lm


@pytest.fixture(autouse=True)
def generation_seed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield


class _CacheTransitionLM(nn.Module):
    """Fixed 0 -> 1 -> ... -> 5 -> 0 transitions with public cache writes.

    Each layer stores the consumed token IDs as scalar keys, offset by
    100 * layer_idx; values are -key - 1. This tests the decoder's calls and
    cache lifecycle without duplicating attention or generation logic.
    """

    def __init__(self, context_length=16):
        super().__init__()
        self.context_length = context_length
        self.num_layers = 2
        self.num_heads = 1
        self.d_head = 1
        # Integer bookkeeping must not determine the KV-cache storage dtype.
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))
        logits = torch.full((6, 6), -torch.inf)
        logits[torch.arange(6), torch.tensor([1, 2, 3, 4, 5, 0])] = 0.0
        self.register_buffer("logits", logits)
        self.inputs = []
        self.caches = []
        self.lengths_before = []
        self.grad_enabled = []
        self.training_flags = []

    @property
    def device(self):
        return self.logits.device

    @property
    def dtype(self):
        return self.logits.dtype

    def forward(self, token_ids, kv_cache=None):
        assert token_ids.shape[-1] <= self.context_length
        self.inputs.append(token_ids.detach().clone())
        self.caches.append(kv_cache)
        self.grad_enabled.append(torch.is_grad_enabled())
        self.training_flags.append(self.training)
        if kv_cache is not None:
            self.lengths_before.append(tuple(kv_cache.get_length(layer) for layer in range(self.num_layers)))
            for layer in range(self.num_layers):
                keys = token_ids[:, None, :, None].to(self.dtype) + 100 * layer
                kv_cache.append(layer, keys, -keys - 1)
        return self.logits[token_ids]


@pytest.fixture
def kv_generation_model():
    return _CacheTransitionLM()


def test_generate_kv_prefill_and_decode_helpers(kv_generation_model):
    model = kv_generation_model.eval()
    prompt = torch.tensor([[0, 1], [1, 2]])

    with torch.enable_grad():
        first, cache = run_prefill(model, prompt)

    torch.testing.assert_close(first, torch.tensor([[2], [3]]))
    assert [cache.get_length(layer) for layer in range(model.num_layers)] == [2, 2]

    with torch.enable_grad():
        second = run_decode_step(model, first, cache)

    torch.testing.assert_close(second, torch.tensor([[3], [4]]))
    torch.testing.assert_close(model.inputs[0], prompt)
    torch.testing.assert_close(model.inputs[1], first)
    assert all(item is cache for item in model.caches)
    assert [cache.get_length(layer) for layer in range(model.num_layers)] == [3, 3]
    assert model.grad_enabled == [False, False]


def test_generate_kv_prefill_rejects_overlong_prompt():
    model = _CacheTransitionLM(context_length=2)

    with pytest.raises(ValueError):
        run_prefill(model, torch.tensor([[0, 1, 2]]))

    assert model.inputs == []


@pytest.mark.parametrize("tokens", [torch.tensor([2]), torch.tensor([[2, 3]])])
def test_generate_kv_decode_step_requires_one_token(kv_generation_model, tokens):
    _, cache = run_prefill(kv_generation_model, torch.tensor([[0, 1]]))

    with pytest.raises(ValueError):
        run_decode_step(kv_generation_model, tokens, cache)

    assert len(kv_generation_model.inputs) == 1
    assert [cache.get_length(layer) for layer in range(kv_generation_model.num_layers)] == [2, 2]


def test_generate_kv_prefill_uses_live_model_dtype(ts_state_dict, in_indices):
    weights, source_config = ts_state_dict
    config = {
        name: source_config[name]
        for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")
    }
    # Moving the model can make stored constructor dtype metadata stale.
    model = get_transformer_lm(config, weights).to(dtype=torch.float64)
    prompt = in_indices[:, :2]
    with torch.no_grad():
        expected = model(prompt)[:, -1, :].argmax(dim=-1, keepdim=True)

    actual, cache = run_prefill(model, prompt, top_p=1e-6)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    reference = next(model.parameters())
    for layer in range(config["num_layers"]):
        assert cache.get_length(layer) == 2
        for tensor in cache.get(layer):
            assert tensor.dtype == reference.dtype
            assert tensor.device == reference.device


def _assert_consumed_length(model, expected_length):
    assert model.caches
    cache = model.caches[-1]
    assert cache is not None
    assert all(item is cache for item in model.caches)
    for layer in range(model.num_layers):
        assert cache.get_length(layer) == expected_length


@pytest.mark.parametrize("max_new_tokens", [0, 1, 4, 7])
def test_generate_kv_token_limit(kv_generation_model, max_new_tokens):
    prompt = torch.tensor([[0, 1]])
    expected = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2]])[:, : 2 + max_new_tokens]

    actual = run_generate_kv_with_model(kv_generation_model, prompt, max_new_tokens=max_new_tokens)

    torch.testing.assert_close(actual, expected)
    assert len(kv_generation_model.inputs) == max_new_tokens
    if max_new_tokens:
        _assert_consumed_length(kv_generation_model, expected_length=1 + max_new_tokens)


def test_generate_kv_prefill_then_single_token_decode(kv_generation_model):
    prompt = torch.tensor([[0, 1], [1, 2]])
    expected = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    actual = run_generate_kv_with_model(kv_generation_model, prompt, max_new_tokens=3)

    torch.testing.assert_close(actual, expected)
    assert len(kv_generation_model.inputs) == 3
    torch.testing.assert_close(kv_generation_model.inputs[0], prompt)
    torch.testing.assert_close(kv_generation_model.inputs[1], expected[:, 2:3])
    torch.testing.assert_close(kv_generation_model.inputs[2], expected[:, 3:4])
    assert kv_generation_model.lengths_before == [(0, 0), (2, 2), (3, 3)]
    _assert_consumed_length(kv_generation_model, expected_length=4)
    cache = kv_generation_model.caches[-1]
    for layer in range(kv_generation_model.num_layers):
        keys, values = cache.get(layer)
        expected_keys = expected[:, None, :-1, None].to(kv_generation_model.dtype) + 100 * layer
        assert keys.shape == (2, 1, 4, 1)
        assert keys.dtype == kv_generation_model.dtype
        assert keys.device == prompt.device
        torch.testing.assert_close(keys, expected_keys, atol=0, rtol=0)
        torch.testing.assert_close(values, -expected_keys - 1, atol=0, rtol=0)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_generate_kv_batch_independence(batch_size):
    model = _CacheTransitionLM()
    prompt = torch.arange(batch_size).reshape(-1, 1)
    expected = torch.cat((prompt, (prompt + 1) % 6, (prompt + 2) % 6, (prompt + 3) % 6), dim=1)

    actual = run_generate_kv_with_model(model, prompt, max_new_tokens=3)

    torch.testing.assert_close(actual, expected)
    _assert_consumed_length(model, expected_length=3)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_generate_kv_preserves_prompt(kv_generation_model, dtype):
    prompt = torch.tensor([[0, 1], [1, 2]], dtype=dtype)
    original = prompt.clone()

    actual = run_generate_kv_with_model(kv_generation_model, prompt, max_new_tokens=2)

    torch.testing.assert_close(prompt, original, atol=0, rtol=0)
    torch.testing.assert_close(actual[:, :2], original, atol=0, rtol=0)
    assert actual.dtype == prompt.dtype
    assert actual.device == prompt.device


@pytest.mark.parametrize("context_length", [2, 3, 4])
def test_generate_kv_context_length(context_length):
    model = _CacheTransitionLM(context_length=context_length)
    prompt = torch.tensor([[0, 1]])
    expected = torch.tensor([[0, 1, 2, 3, 4]])[:, : context_length + 1]

    actual = run_generate_kv_with_model(model, prompt, max_new_tokens=7)

    # A model call at the context limit may produce one final output token.
    torch.testing.assert_close(actual, expected)
    assert len(model.inputs) == context_length - 1
    _assert_consumed_length(model, expected_length=context_length)


@pytest.mark.parametrize("prompt, expected", [([[3, 4]], [[3, 4, 5]]), ([[0, 1]], [[0, 1, 2, 3, 4, 5]])])
def test_generate_kv_stops_at_eos(prompt, expected):
    model = _CacheTransitionLM()
    prompt = torch.tensor(prompt)

    actual = run_generate_kv_with_model(model, prompt, max_new_tokens=8, eos_token_id=5)

    torch.testing.assert_close(actual, torch.tensor(expected))
    assert len(model.inputs) == len(expected[0]) - prompt.shape[-1]
    _assert_consumed_length(model, expected_length=len(expected[0]) - 1)


def test_generate_kv_prompt_exceeds_context_length():
    model = _CacheTransitionLM(context_length=2)
    prompt = torch.tensor([[0, 1, 2]])

    actual = run_generate_kv_with_model(model, prompt, max_new_tokens=3)

    torch.testing.assert_close(actual, prompt, atol=0, rtol=0)
    assert model.inputs == []


@pytest.mark.parametrize("max_new_tokens", [3, 8])
def test_generate_kv_batch_eos(kv_generation_model, max_new_tokens):
    prompt = torch.tensor([[3], [0]])
    expected = torch.tensor([[3, 4, 5, 5, 5, 5], [0, 1, 2, 3, 4, 5]])[:, : 1 + max_new_tokens]

    actual = run_generate_kv_with_model(kv_generation_model, prompt, max_new_tokens=max_new_tokens, eos_token_id=5)

    torch.testing.assert_close(actual, expected)
    assert len(kv_generation_model.inputs) == min(5, max_new_tokens)
    _assert_consumed_length(kv_generation_model, expected_length=expected.shape[-1] - 1)


@pytest.mark.parametrize("eos_token_id, expected", [(None, [[4, 5, 0, 1, 2]]), (0, [[4, 5, 0]])])
def test_generate_kv_eos_disabled_or_zero(kv_generation_model, eos_token_id, expected):
    actual = run_generate_kv_with_model(
        kv_generation_model, torch.tensor([[4]]), max_new_tokens=4, eos_token_id=eos_token_id
    )

    torch.testing.assert_close(actual, torch.tensor(expected))
    _assert_consumed_length(kv_generation_model, expected_length=len(expected[0]) - 1)


def test_generate_kv_forwards_sampling_options(kv_generation_model, monkeypatch):
    from cs336_basics import decoding

    calls = []

    def record_sample(logits, temperature=1.0, top_p=1.0):
        calls.append((logits.detach().clone(), temperature, top_p))
        return torch.ones((logits.shape[0], 1), dtype=torch.long, device=logits.device)

    monkeypatch.setattr(decoding, "sample_next_token", record_sample)
    actual = run_generate_kv_with_model(
        kv_generation_model, torch.tensor([[0, 1]]), max_new_tokens=2, temperature=0.7, top_p=0.8
    )

    torch.testing.assert_close(actual, torch.tensor([[0, 1, 1, 1]]))
    assert len(calls) == 2
    for logits, temperature, top_p in calls:
        torch.testing.assert_close(logits, kv_generation_model.logits[torch.tensor([1])])
        assert temperature == 0.7
        assert top_p == 0.8


def test_generate_kv_eval_and_no_grad(kv_generation_model):
    kv_generation_model.train()

    with torch.enable_grad():
        run_generate_kv_with_model(kv_generation_model, torch.tensor([[0, 1]]), max_new_tokens=2)

    assert kv_generation_model.training_flags == [False, False]
    assert kv_generation_model.grad_enabled == [False, False]


def test_generate_kv_fresh_cache_per_call(kv_generation_model):
    first = run_generate_kv_with_model(kv_generation_model, torch.tensor([[0, 1]]), max_new_tokens=2)
    first_cache = kv_generation_model.caches[0]
    first_contents = [tuple(tensor.clone() for tensor in first_cache.get(layer)) for layer in range(2)]

    second = run_generate_kv_with_model(kv_generation_model, torch.tensor([[3, 4]]), max_new_tokens=2)

    torch.testing.assert_close(first, torch.tensor([[0, 1, 2, 3]]))
    torch.testing.assert_close(second, torch.tensor([[3, 4, 5, 0]]))
    assert len(kv_generation_model.caches) == 4
    assert kv_generation_model.caches[0] is kv_generation_model.caches[1]
    assert kv_generation_model.caches[2] is kv_generation_model.caches[3]
    assert kv_generation_model.caches[2] is not first_cache
    assert kv_generation_model.lengths_before[2] == (0, 0)
    for layer, pair in enumerate(first_contents):
        for actual, expected in zip(first_cache.get(layer), pair, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("context_length", [None, 8.0, "8"])
def test_generate_kv_requires_integer_context_length(context_length):
    model = _CacheTransitionLM(context_length=context_length)

    with pytest.raises(TypeError):
        run_generate_kv_with_model(model, torch.tensor([[0, 1]]), max_new_tokens=2)

    assert model.inputs == []


@pytest.mark.parametrize("max_new_tokens", [0, 1, 3, 8])
def test_generate_kv_matches_naive_from_weights(
    ts_state_dict, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, in_indices, max_new_tokens
):
    weights, _ = ts_state_dict
    model_args = dict(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=weights,
    )
    original = in_indices.clone()
    # Below 1 / vocab_size, nucleus sampling retains only the largest logit.
    expected = run_generate_naive(**model_args, prompt_ids=in_indices, max_new_tokens=max_new_tokens, top_p=1e-6)

    actual = run_generate_kv(**model_args, prompt_ids=in_indices, max_new_tokens=max_new_tokens, top_p=1e-6)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(in_indices, original, atol=0, rtol=0)
