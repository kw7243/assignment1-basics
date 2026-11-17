"""Test the KVCache contract through its public operations.

Testing strategy:
    Constructor/get/get_length: singleton and larger dimensions; empty,
        partial, and full prefixes; configured dtype and device.
    Append: one write and multiple appends; zero, one, and multiple tokens;
        exact capacity; distinct batch/head/layer data; noncontiguous inputs.
    Operation sequences: interleaved layer writes; independent instances;
        repeated observations; clear from empty, partial, and full states;
        repeated clear and reuse.
    Aliasing: append copies input values; returned views are read-only to
        clients; clear preserves storage as specified.

Invalid-argument behavior is not specified by the current contract. Tests
respect append's preconditions and do not require particular exceptions.
"""

import math

import pytest
import torch

from .adapters import get_kv_cache


@pytest.fixture
def kv_cache_config():
    return dict(
        num_layers=3,
        batch_size=2,
        num_kv_heads=2,
        max_seq_len=5,
        d_head=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


@pytest.fixture
def kv_cache(kv_cache_config):
    return get_kv_cache(**kv_cache_config)


def _kv_block(config, length, offset=0):
    shape = (config["batch_size"], config["num_kv_heads"], length, config["d_head"])
    keys = torch.arange(math.prod(shape), device=config["device"], dtype=config["dtype"]).reshape(shape) + offset
    values = -keys - 1
    return keys, values


def _assert_prefix(cache, layer_idx, expected_keys, expected_values, capacity):
    length = cache.get_length(layer_idx)
    assert isinstance(length, int)
    assert 0 <= length <= capacity
    assert length == expected_keys.shape[2]
    keys, values = cache.get(layer_idx)
    for actual, expected in ((keys, expected_keys), (values, expected_values)):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dimensions", [(1, 1, 1, 1, 1), (3, 2, 2, 5, 3)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_kv_cache_initial_state(kv_cache_config, dimensions, dtype):
    config = kv_cache_config | dict(
        zip(("num_layers", "batch_size", "num_kv_heads", "max_seq_len", "d_head"), dimensions, strict=True)
    )
    config["dtype"] = dtype
    cache = get_kv_cache(**config)
    empty_keys, empty_values = _kv_block(config, 0)

    for layer in range(config["num_layers"]):
        _assert_prefix(cache, layer, empty_keys, empty_values, config["max_seq_len"])


@pytest.mark.parametrize("chunk_lengths", [(5,), (1, 4), (2, 1, 2), (1, 1, 1, 1, 1)])
def test_kv_cache_append_preserves_prefix(kv_cache, kv_cache_config, chunk_lengths):
    keys, values = _kv_block(kv_cache_config, 5)
    length = 0

    for chunk_length in chunk_lengths:
        end = length + chunk_length
        kv_cache.append(0, keys[:, :, length:end, :], values[:, :, length:end, :])
        length = end
        _assert_prefix(kv_cache, 0, keys[:, :, :length, :], values[:, :, :length, :], 5)


def test_kv_cache_layers_are_independent(kv_cache, kv_cache_config):
    # First, middle, and last layers advance at different rates and in a
    # different order. Check every layer after each mutation.
    expected = [_kv_block(kv_cache_config, 0) for _ in range(3)]
    writes = [(2, 2), (0, 1), (2, 1), (1, 5), (0, 3), (2, 2)]

    for step, (layer, chunk_length) in enumerate(writes):
        new_keys, new_values = _kv_block(kv_cache_config, chunk_length, offset=100 * (step + 1))
        previous_keys, previous_values = expected[layer]
        kv_cache.append(layer, new_keys, new_values)
        expected[layer] = (
            torch.cat((previous_keys, new_keys), dim=2),
            torch.cat((previous_values, new_values), dim=2),
        )
        for observed_layer, (keys, values) in enumerate(expected):
            _assert_prefix(kv_cache, observed_layer, keys, values, 5)


@pytest.mark.parametrize("length", [0, 2, 5])
def test_kv_cache_empty_append_preserves_state(kv_cache, kv_cache_config, length):
    keys, values = _kv_block(kv_cache_config, length)
    if length:
        kv_cache.append(0, keys, values)
    empty_keys, empty_values = _kv_block(kv_cache_config, 0)

    kv_cache.append(0, empty_keys, empty_values)

    _assert_prefix(kv_cache, 0, keys, values, 5)
    for layer in (1, 2):
        _assert_prefix(kv_cache, layer, empty_keys, empty_values, 5)


def test_kv_cache_observers_preserve_state(kv_cache, kv_cache_config):
    keys, values = _kv_block(kv_cache_config, 2)
    kv_cache.append(1, keys, values)

    for _ in range(3):
        _assert_prefix(kv_cache, 1, keys, values, 5)

    new_keys, new_values = _kv_block(kv_cache_config, 1, offset=100)
    kv_cache.append(1, new_keys, new_values)
    _assert_prefix(kv_cache, 1, torch.cat((keys, new_keys), dim=2), torch.cat((values, new_values), dim=2), 5)


def test_kv_cache_append_copies_input_values(kv_cache, kv_cache_config):
    keys, values = _kv_block(kv_cache_config, 2)
    expected_keys, expected_values = keys.clone(), values.clone()

    kv_cache.append(0, keys, values)

    # append must leave its inputs unchanged.
    torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
    # Later changes to caller-owned tensors do not change the written prefix.
    keys.fill_(999)
    values.fill_(999)
    _assert_prefix(kv_cache, 0, expected_keys, expected_values, 5)


def test_kv_cache_noncontiguous_append(kv_cache, kv_cache_config):
    wide_config = kv_cache_config | {"d_head": 2 * kv_cache_config["d_head"]}
    wide_keys, wide_values = _kv_block(wide_config, 3)
    keys, values = wide_keys[..., ::2], wide_values[..., ::2]
    assert not keys.is_contiguous()
    assert not values.is_contiguous()
    first_keys, first_values = _kv_block(kv_cache_config, 1, offset=100)
    kv_cache.append(0, first_keys, first_values)

    kv_cache.append(0, keys, values)

    _assert_prefix(kv_cache, 0, torch.cat((first_keys, keys), dim=2), torch.cat((first_values, values), dim=2), 5)


@pytest.mark.parametrize("initial_lengths", [(0, 0, 0), (2, 0, 1), (5, 5, 5)])
def test_kv_cache_clear_and_reuse(kv_cache, kv_cache_config, initial_lengths):
    for layer, length in enumerate(initial_lengths):
        if length:
            keys, values = _kv_block(kv_cache_config, length, offset=100 * layer)
            kv_cache.append(layer, keys, values)

    empty_keys, empty_values = _kv_block(kv_cache_config, 0)
    for _ in range(2):
        kv_cache.clear()
        for layer in range(3):
            _assert_prefix(kv_cache, layer, empty_keys, empty_values, 5)

    # The new prefix is shorter than the previous full-cache case. Old tokens
    # outside the new prefix must not become visible through get().
    for layer in range(3):
        keys, values = _kv_block(kv_cache_config, 2, offset=1000 + 100 * layer)
        kv_cache.append(layer, keys, values)
        _assert_prefix(kv_cache, layer, keys, values, 5)


def test_kv_cache_clear_preserves_storage(kv_cache, kv_cache_config):
    views_before_clear = []
    for layer in range(3):
        keys, values = _kv_block(kv_cache_config, 5, offset=100 * layer)
        kv_cache.append(layer, keys, values)
        views_before_clear.append(kv_cache.get(layer))

    kv_cache.clear()

    for layer, (old_keys, old_values) in enumerate(views_before_clear):
        keys, values = _kv_block(kv_cache_config, 2, offset=1000 + 100 * layer)
        kv_cache.append(layer, keys, values)
        _assert_prefix(kv_cache, layer, keys, values, 5)
        new_keys, new_values = kv_cache.get(layer)
        # This checks the explicit storage-reuse contract through public
        # tensor views. Keeping the old views alive prevents address reuse.
        assert new_keys.data_ptr() == old_keys.data_ptr()
        assert new_values.data_ptr() == old_values.data_ptr()
        torch.testing.assert_close(old_keys[:, :, :2, :], keys, rtol=0, atol=0)
        torch.testing.assert_close(old_values[:, :, :2, :], values, rtol=0, atol=0)


def test_kv_cache_instances_are_independent(kv_cache, kv_cache_config):
    other = get_kv_cache(**kv_cache_config)
    keys, values = _kv_block(kv_cache_config, 2)
    other_keys, other_values = _kv_block(kv_cache_config, 3, offset=100)
    empty_keys, empty_values = _kv_block(kv_cache_config, 0)

    kv_cache.append(0, keys, values)
    _assert_prefix(other, 0, empty_keys, empty_values, 5)
    other.append(0, other_keys, other_values)
    kv_cache.clear()

    _assert_prefix(kv_cache, 0, empty_keys, empty_values, 5)
    _assert_prefix(other, 0, other_keys, other_values, 5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_kv_cache_append_preserves_dtype(kv_cache_config, dtype):
    config = kv_cache_config | {"dtype": dtype}
    cache = get_kv_cache(**config)
    keys, values = _kv_block(config, 5)

    cache.append(0, keys, values)

    _assert_prefix(cache, 0, keys, values, 5)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
        ),
    ],
)
def test_kv_cache_append_preserves_device(kv_cache_config, device):
    config = kv_cache_config | {"device": torch.device(device)}
    cache = get_kv_cache(**config)
    keys, values = _kv_block(config, 5)

    cache.append(0, keys, values)

    _assert_prefix(cache, 0, keys, values, 5)
