"""Integration tests for externally owned Transformer KV caches.

Use public constructors, state dictionaries, forward calls, and cache observers.
Partition by layers, batch size, prefix length, chunk size, and exact capacity.
Compare logits with tolerances and greedy token IDs exactly. Fixed attention
examples supply independent oracles for K/V contents, RoPE, and causality.

Cached masks use (query_length, cached_length + query_length) dimensions;
True permits a key, and the causal constraint still applies. Invalid-input
tests use the documented dimension and input-width ValueError contracts.
"""

import math

import pytest
import torch

from .kv_cache_integration_adapters import (
    get_integration_cache,
    get_self_attention,
    get_transformer_block,
    get_transformer_lm,
)


@pytest.fixture(autouse=True)
def integration_inference_mode():
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(42)
        yield


@pytest.fixture
def model_config():
    return dict(vocab_size=17, context_length=8, d_model=8, num_layers=3, num_heads=2, d_ff=12, rope_theta=10_000.0)


def _model_weights(config):
    # Public state-dict names and shapes match the existing Transformer adapter.
    generator = torch.Generator().manual_seed(7)

    def weight(*shape):
        return 0.2 * torch.randn(shape, generator=generator)

    d_model, d_ff = config["d_model"], config["d_ff"]
    weights = {
        "token_embeddings.weight": weight(config["vocab_size"], d_model),
        "ln_final.weight": torch.ones(d_model),
        "lm_head.weight": weight(config["vocab_size"], d_model),
    }
    for layer in range(config["num_layers"]):
        prefix = f"layers.{layer}."
        weights[prefix + "ln1.weight"] = torch.ones(d_model)
        weights[prefix + "ln2.weight"] = torch.ones(d_model)
        for projection in ("q", "k", "v", "output"):
            weights[prefix + f"attn.{projection}_proj.weight"] = weight(d_model, d_model)
        weights[prefix + "ffn.w1.weight"] = weight(d_ff, d_model)
        weights[prefix + "ffn.w2.weight"] = weight(d_model, d_ff)
        weights[prefix + "ffn.w3.weight"] = weight(d_ff, d_model)
    return weights


@pytest.fixture
def model_weights(model_config):
    return _model_weights(model_config)


@pytest.fixture
def model(model_config, model_weights):
    return get_transformer_lm(model_config, model_weights)


def _token_ids(config, batch_size, length, offset=0):
    return (torch.arange(batch_size * length).reshape(batch_size, length) * 3 + offset) % config["vocab_size"]


def _assert_cache(cache, config, batch_size, length, active_layers=None):
    if active_layers is None:
        active_layers = range(config["num_layers"])
    for layer in range(config["num_layers"]):
        expected_length = length if layer in active_layers else 0
        assert cache.get_length(layer) == expected_length
        keys, values = cache.get(layer)
        expected_shape = (batch_size, config["num_heads"], expected_length, config["d_model"] // config["num_heads"])
        for tensor in (keys, values):
            assert tensor.shape == expected_shape
            assert tensor.dtype == torch.float32
            assert tensor.device == torch.device("cpu")
            assert torch.isfinite(tensor).all()


def _cache_contents(cache, num_layers):
    # get() returns borrowed views; snapshots must not alias later writes.
    return [tuple(tensor.clone() for tensor in cache.get(layer)) for layer in range(num_layers)]


def _assert_cache_contents(cache, expected, length=None):
    for layer, pair in enumerate(expected):
        for actual, reference in zip(cache.get(layer), pair, strict=True):
            if length is not None:
                actual, reference = actual[:, :, :length, :], reference[:, :, :length, :]
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "num_layers,batch_size,chunks",
    [
        (1, 1, (1,)),
        (1, 2, (3,)),
        (3, 1, (2, 1, 3)),
        (3, 2, (1, 1, 1, 1, 1, 1)),
        (3, 2, (3, 3)),
        (3, 2, (6,)),
        (3, 1, (7, 1)),
        (3, 2, (3, 5)),
    ],
)
def test_kv_cache_integration_prefill_and_decode(model_config, num_layers, batch_size, chunks):
    config = model_config | {"num_layers": num_layers}
    model = get_transformer_lm(config, _model_weights(config))
    tokens = _token_ids(config, batch_size, sum(chunks))
    cache = get_integration_cache(config, batch_size)
    reference_cache = get_integration_cache(config, batch_size)
    expected = model(tokens)
    one_shot = model(tokens, kv_cache=reference_cache)
    _assert_cache(reference_cache, config, batch_size, tokens.shape[-1])
    torch.testing.assert_close(one_shot, expected, atol=1e-5, rtol=1e-4)
    reference_contents = _cache_contents(reference_cache, num_layers)
    _assert_cache(cache, config, batch_size, 0)
    outputs = []
    start = 0

    for chunk_length in chunks:
        previous = _cache_contents(cache, num_layers)
        end = start + chunk_length
        actual = model(tokens[:, start:end], kv_cache=cache)
        assert actual.shape == (batch_size, chunk_length, config["vocab_size"])
        torch.testing.assert_close(actual, expected[:, start:end], atol=1e-5, rtol=1e-4)
        _assert_cache(cache, config, batch_size, end)
        _assert_cache_contents(cache, reference_contents, length=end)
        # Appending must not re-rotate or overwrite existing prefix vectors.
        for layer, pair in enumerate(previous):
            for current, old in zip(cache.get(layer), pair, strict=True):
                torch.testing.assert_close(current[:, :, :start, :], old, atol=0, rtol=0)
        outputs.append(actual)
        start = end

    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-5, rtol=1e-4)


@pytest.fixture
def analytic_attention_case(model_config):
    config = model_config | {"d_model": 4}
    # Q=0 gives uniform attention over permitted keys. Identity V/output
    # projections make each output the arithmetic mean of its allowed inputs.
    weights = {
        "q_proj.weight": torch.zeros(4, 4),
        "k_proj.weight": torch.eye(4),
        "v_proj.weight": torch.eye(4),
        "output_proj.weight": torch.eye(4),
    }
    inputs = torch.tensor(
        [[[1.0, 0.0, 10.0, 0.0], [0.0, 1.0, 0.0, 20.0], [2.0, 0.0, 30.0, 0.0], [0.0, 3.0, 0.0, 40.0]]]
    )
    outputs = torch.tensor(
        [[[1.0, 0.0, 10.0, 0.0], [0.5, 0.5, 5.0, 10.0], [1.0, 1 / 3, 40 / 3, 20 / 3], [0.75, 1.0, 10.0, 15.0]]]
    )
    # For d_head=2, RoPE is a 2D rotation by the absolute token position.
    s1, s2, s3 = (math.sin(position) for position in (1, 2, 3))
    c1, c2, c3 = (math.cos(position) for position in (1, 2, 3))
    keys = torch.tensor(
        [
            [
                [[1.0, 0.0], [-s1, c1], [2 * c2, 2 * s2], [-3 * s3, 3 * c3]],
                [[10.0, 0.0], [-20 * s1, 20 * c1], [30 * c2, 30 * s2], [-40 * s3, 40 * c3]],
            ]
        ]
    )
    values = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]], [[10.0, 0.0], [0.0, 20.0], [30.0, 0.0], [0.0, 40.0]]]]
    )
    return config, weights, inputs, outputs, keys, values


@pytest.mark.parametrize(
    "layer_idx,chunks,use_rope",
    [(0, (4,), True), (1, (2, 2), True), (2, (1, 1, 1, 1), True), (1, (2, 2), False)],
)
def test_kv_cache_integration_attention_contents_and_rope(analytic_attention_case, layer_idx, chunks, use_rope):
    config, weights, inputs, expected, expected_keys, expected_values = analytic_attention_case
    attention = get_self_attention(config, weights, layer_idx, use_rope=use_rope)
    if not use_rope:
        expected_keys = expected_values
    cache = get_integration_cache(config, 1)
    start = 0

    for chunk_length in chunks:
        end = start + chunk_length
        actual = attention(inputs[:, start:end], kv_cache=cache)
        torch.testing.assert_close(actual, expected[:, start:end], atol=1e-5, rtol=1e-4)
        _assert_cache(cache, config, 1, end, active_layers=(layer_idx,))
        keys, values = cache.get(layer_idx)
        torch.testing.assert_close(keys, expected_keys[:, :, :end], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(values, expected_values[:, :, :end], atol=0, rtol=0)
        start = end


@pytest.mark.parametrize("exclude_first_key", [False, True])
def test_kv_cache_integration_rectangular_causal_mask(analytic_attention_case, exclude_first_key):
    config, weights, inputs, expected, _, _ = analytic_attention_case
    attention = get_self_attention(config, weights, layer_idx=1)
    cache = get_integration_cache(config, 1)
    attention(inputs[:, :2], kv_cache=cache)
    mask = torch.ones(2, 4, dtype=torch.bool)
    if exclude_first_key:
        mask[:, 0] = False
        expected = torch.tensor([[[1.0, 0.5, 15.0, 10.0], [2 / 3, 4 / 3, 10.0, 20.0]]])
    else:
        expected = expected[:, 2:]

    actual = attention(inputs[:, 2:], mask=mask, kv_cache=cache)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    _assert_cache(cache, config, 1, 4, active_layers=(1,))


@pytest.mark.parametrize("layer_idx", [0, 1, 2])
def test_kv_cache_integration_transformer_block(model_config, model_weights, layer_idx):
    prefix = f"layers.{layer_idx}."
    weights = {name.removeprefix(prefix): value for name, value in model_weights.items() if name.startswith(prefix)}
    block = get_transformer_block(model_config, weights, layer_idx)
    inputs = torch.randn(2, 6, model_config["d_model"], generator=torch.Generator().manual_seed(9))
    # Block forwards may modify their input, so each call owns its tensor.
    expected = block(inputs.clone())
    cache = get_integration_cache(model_config, 2)
    start = 0

    for chunk_length in (2, 1, 3):
        end = start + chunk_length
        actual = block(inputs[:, start:end].clone(), kv_cache=cache)
        torch.testing.assert_close(actual, expected[:, start:end], atol=1e-5, rtol=1e-4)
        _assert_cache(cache, model_config, 2, end, active_layers=(layer_idx,))
        start = end


def test_kv_cache_integration_absolute_positions(model, model_config):
    tokens = _token_ids(model_config, 2, 6)
    implicit_cache = get_integration_cache(model_config, 2)
    explicit_cache = get_integration_cache(model_config, 2)
    expected = model(tokens, token_positions=torch.arange(6).expand(2, 6))
    start = 0

    for chunk_length in (2, 1, 3):
        end = start + chunk_length
        chunk = tokens[:, start:end]
        implicit = model(chunk, kv_cache=implicit_cache)
        explicit = model(
            chunk, token_positions=torch.arange(start, end).expand(2, chunk_length), kv_cache=explicit_cache
        )
        torch.testing.assert_close(implicit, expected[:, start:end], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(implicit, explicit, atol=1e-5, rtol=1e-4)
        _assert_cache(implicit_cache, model_config, 2, end)
        _assert_cache_contents(implicit_cache, _cache_contents(explicit_cache, model_config["num_layers"]))
        start = end


@pytest.mark.parametrize("prefill_length", [0, 2])
def test_kv_cache_integration_future_tokens_are_inaccessible(model, model_config, prefill_length):
    tokens = _token_ids(model_config, 2, 6)
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % model_config["vocab_size"]
    first_cache = get_integration_cache(model_config, 2)
    second_cache = get_integration_cache(model_config, 2)
    if prefill_length:
        model(tokens[:, :prefill_length], kv_cache=first_cache)
        model(changed[:, :prefill_length], kv_cache=second_cache)

    first = model(tokens[:, prefill_length:], kv_cache=first_cache)
    second = model(changed[:, prefill_length:], kv_cache=second_cache)

    torch.testing.assert_close(first[:, : 4 - prefill_length], second[:, : 4 - prefill_length], atol=1e-5, rtol=1e-4)
    _assert_cache(first_cache, model_config, 2, 6)
    _assert_cache(second_cache, model_config, 2, 6)
    _assert_cache_contents(first_cache, _cache_contents(second_cache, model_config["num_layers"]), length=4)


@pytest.mark.parametrize("batch_size,prompt_length", [(1, 1), (1, 3), (2, 1), (2, 3)])
def test_kv_cache_integration_greedy_token_ids(model, model_config, batch_size, prompt_length):
    tokens = _token_ids(model_config, batch_size, prompt_length)
    cache = get_integration_cache(model_config, batch_size)
    cached_logits = model(tokens, kv_cache=cache)[:, -1, :]

    for step in range(3):
        expected = model(tokens)[:, -1, :]
        # The fixture must have a unique maximum, with room for rounding.
        top_two = expected.topk(2, dim=-1).values
        assert ((top_two[:, 0] - top_two[:, 1]) > 1e-5).all()
        torch.testing.assert_close(cached_logits, expected, atol=1e-5, rtol=1e-4)
        actual_token = cached_logits.argmax(dim=-1, keepdim=True)
        expected_token = expected.argmax(dim=-1, keepdim=True)
        torch.testing.assert_close(actual_token, expected_token, atol=0, rtol=0)
        _assert_cache(cache, model_config, batch_size, prompt_length + step)
        tokens = torch.cat((tokens, actual_token), dim=1)
        if step < 2:
            cached_logits = model(actual_token, kv_cache=cache)[:, -1, :]


def test_kv_cache_integration_clear_and_reuse(model, model_config):
    cache = get_integration_cache(model_config, 2)
    model(_token_ids(model_config, 2, 6), kv_cache=cache)
    _assert_cache(cache, model_config, 2, 6)

    cache.clear()

    _assert_cache(cache, model_config, 2, 0)
    tokens = _token_ids(model_config, 2, 4, offset=5)
    fresh = get_integration_cache(model_config, 2)
    expected = model(tokens)
    fresh_logits = model(tokens, kv_cache=fresh)
    first = model(tokens[:, :3], kv_cache=cache)
    last = model(tokens[:, 3:], kv_cache=cache)
    actual = torch.cat((first, last), dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(actual, fresh_logits, atol=1e-5, rtol=1e-4)
    _assert_cache(cache, model_config, 2, 4)
    _assert_cache_contents(cache, _cache_contents(fresh, model_config["num_layers"]))


def test_kv_cache_integration_external_cache_ownership(model, model_config):
    first_tokens = _token_ids(model_config, 2, 4)
    second_tokens = _token_ids(model_config, 2, 5, offset=7)
    first_cache = get_integration_cache(model_config, 2)
    second_cache = get_integration_cache(model_config, 2)
    uncached_before = model(first_tokens)
    model(first_tokens[:, :2], kv_cache=first_cache)
    first_snapshot = _cache_contents(first_cache, model_config["num_layers"])
    model(second_tokens[:, :3], kv_cache=second_cache)
    _assert_cache_contents(first_cache, first_snapshot)
    second_snapshot = _cache_contents(second_cache, model_config["num_layers"])

    first = model(first_tokens[:, 2:], kv_cache=first_cache)

    torch.testing.assert_close(first, uncached_before[:, 2:], atol=1e-5, rtol=1e-4)
    _assert_cache_contents(second_cache, second_snapshot)
    _assert_cache(first_cache, model_config, 2, 4)
    _assert_cache(second_cache, model_config, 2, 3)
    first_snapshot = _cache_contents(first_cache, model_config["num_layers"])
    second = model(second_tokens[:, 3:], kv_cache=second_cache)
    torch.testing.assert_close(second, model(second_tokens)[:, 3:], atol=1e-5, rtol=1e-4)
    _assert_cache_contents(first_cache, first_snapshot)
    torch.testing.assert_close(model(first_tokens), uncached_before, atol=0, rtol=0)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_kv_cache_integration_uncached_reference(ts_state_dict, in_indices, numpy_snapshot, explicit_none):
    weights, source_config = ts_state_dict
    config = {
        name: source_config[name]
        for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")
    }
    model = get_transformer_lm(config, weights)

    actual = model(in_indices, kv_cache=None) if explicit_none else model(in_indices)

    numpy_snapshot.assert_match(actual, test_name="test_transformer_lm", atol=1e-4, rtol=1e-2)


@pytest.mark.parametrize("component", ["attention", "block"])
@pytest.mark.parametrize("d_model,num_heads", [(0, 2), (8, 0), (7, 2)])
def test_kv_cache_integration_invalid_dimensions(model_config, component, d_model, num_heads):
    config = model_config | {"d_model": d_model, "num_heads": num_heads}
    if component == "attention":
        from cs336_basics.attention import MultiHeadSelfAttention

        constructor = MultiHeadSelfAttention
        kwargs = {}
    else:
        from cs336_basics.transformer import TransformerBlock

        constructor = TransformerBlock
        kwargs = {"d_ff": config["d_ff"]}

    with pytest.raises(ValueError):
        constructor(d_model=d_model, num_heads=num_heads, pos_layer=None, layer_idx=0, **kwargs)


@pytest.mark.parametrize("width", [3, 5])
def test_kv_cache_integration_invalid_attention_width(analytic_attention_case, width):
    config, weights, _, _, _, _ = analytic_attention_case
    attention = get_self_attention(config, weights, layer_idx=1)
    cache = get_integration_cache(config, 1)

    with pytest.raises(ValueError):
        attention(torch.zeros(1, 2, width), kv_cache=cache)
