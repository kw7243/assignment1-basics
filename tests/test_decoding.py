import math

import pytest
import torch
import torch.nn as nn

from .adapters import run_generate_naive, run_generate_naive_with_model, run_sample_next_token, run_transformer_lm


@pytest.fixture(autouse=True)
def decoding_seed():
    # Restore the CPU RNG state after each test.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield


@pytest.fixture
def decoding_logits():
    # Unsorted probabilities, with different token orders in each batch row.
    return torch.tensor([[0.1, 0.4, 0.2, 0.3], [0.3, 0.2, 0.4, 0.1]], dtype=torch.float64).log()


class _TransitionLM(nn.Module):
    """Test model with the fixed transitions 0 -> 1 -> ... -> 5 -> 0."""

    def __init__(self, context_length=16):
        super().__init__()
        self.context_length = context_length
        logits = torch.full((6, 6), -torch.inf)
        logits[torch.arange(6), torch.tensor([1, 2, 3, 4, 5, 0])] = 0.0
        self.register_buffer("logits", logits)
        self.inputs = []
        self.grad_enabled = []

    def forward(self, token_ids):
        assert token_ids.shape[-1] <= self.context_length, "Model input exceeds the context length"
        self.inputs.append(token_ids.detach().clone())
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.logits[token_ids]


@pytest.fixture
def decoding_model():
    return _TransitionLM()


def _sample_frequencies(logits, **kwargs):
    # Independent draws in one batch keep these distribution checks fast.
    # The fixed seed and tolerance avoid dependence on exact sampled sequences.
    n_samples = 16_384
    samples = run_sample_next_token(logits.repeat_interleave(n_samples, dim=0), **kwargs)
    assert samples.shape == (logits.shape[0] * n_samples, 1)
    assert samples.dtype == torch.long
    assert samples.device == logits.device
    assert ((samples >= 0) & (samples < logits.shape[-1])).all()
    samples = samples.reshape(logits.shape[0], n_samples)
    return torch.nn.functional.one_hot(samples, num_classes=logits.shape[-1]).to(torch.float64).mean(dim=1)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decoding_sample_shape_and_batch_independence(batch_size, dtype):
    logits = torch.full((batch_size, 4), -torch.inf, dtype=dtype)
    expected = torch.arange(batch_size).reshape(-1, 1)
    logits[torch.arange(batch_size), expected.squeeze(-1)] = 0.0

    actual = run_sample_next_token(logits)

    assert actual.shape == (batch_size, 1)
    assert actual.dtype == torch.long
    assert actual.device == logits.device
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_decoding_temperature_distribution(decoding_logits, temperature):
    expected = torch.softmax(decoding_logits / temperature, dim=-1)
    actual = _sample_frequencies(decoding_logits, temperature=temperature, top_p=1.0)

    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0)


@pytest.mark.parametrize("offset", [-10_000.0, 10_000.0])
def test_decoding_large_logits(decoding_logits, offset):
    actual = _sample_frequencies(decoding_logits + offset)

    torch.testing.assert_close(actual, decoding_logits.exp(), atol=0.02, rtol=0)


@pytest.mark.parametrize(
    "top_p, expected",
    [
        (0.01, [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        (0.6, [[0.0, 4 / 7, 0.0, 3 / 7], [3 / 7, 0.0, 4 / 7, 0.0]]),
        (0.8, [[0.0, 4 / 9, 2 / 9, 3 / 9], [3 / 9, 2 / 9, 4 / 9, 0.0]]),
        (0.95, [[0.1, 0.4, 0.2, 0.3], [0.3, 0.2, 0.4, 0.1]]),
        (1.0, [[0.1, 0.4, 0.2, 0.3], [0.3, 0.2, 0.4, 0.1]]),
    ],
)
def test_decoding_top_p_distribution(decoding_logits, top_p, expected):
    expected = torch.tensor(expected, dtype=torch.float64)
    actual = _sample_frequencies(decoding_logits, top_p=top_p)

    # Excluded tokens must never occur, even within the frequency tolerance.
    assert (actual[expected == 0] == 0).all()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0)


def test_decoding_top_p_exact_threshold():
    # Uniform probabilities are exactly representable. Either tied pair is valid.
    actual = _sample_frequencies(torch.zeros(1, 4, dtype=torch.float64), top_p=0.5)

    assert (actual > 0).sum().item() == 2
    torch.testing.assert_close(actual[actual > 0], torch.full((2,), 0.5, dtype=torch.float64), atol=0.02, rtol=0)


def test_decoding_top_p_different_nucleus_sizes():
    logits = torch.tensor([[0.1, 0.7, 0.05, 0.15], [0.2, 0.3, 0.1, 0.4]], dtype=torch.float64).log()
    expected = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 3 / 7, 0.0, 4 / 7]], dtype=torch.float64)

    actual = _sample_frequencies(logits, top_p=0.6)

    assert (actual[expected == 0] == 0).all()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0)


@pytest.mark.parametrize(
    "temperature, expected",
    [(0.5, [0.0, 1.0, 0.0, 0.0]), (2.0, [0.0, math.sqrt(2) / (1 + math.sqrt(2)), 1 / (1 + math.sqrt(2)), 0.0])],
)
def test_decoding_temperature_before_top_p(temperature, expected):
    logits = torch.tensor([[0.125, 0.5, 0.25, 0.125]], dtype=torch.float64).log()
    expected = torch.tensor([expected], dtype=torch.float64)

    actual = _sample_frequencies(logits, temperature=temperature, top_p=0.6)

    assert (actual[expected == 0] == 0).all()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0)


def test_decoding_sample_preserves_logits(decoding_logits):
    original = decoding_logits.clone()

    run_sample_next_token(decoding_logits, temperature=0.5, top_p=0.6)

    torch.testing.assert_close(decoding_logits, original, rtol=0, atol=0)


@pytest.mark.parametrize("max_new_tokens", [0, 1, 4, 7])
def test_decoding_generation_token_limit(decoding_model, max_new_tokens):
    prompt = torch.tensor([[0, 1]])
    expected = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2]])[:, : 2 + max_new_tokens]

    actual = run_generate_naive_with_model(decoding_model, prompt, max_new_tokens=max_new_tokens)

    torch.testing.assert_close(actual, expected)
    assert len(decoding_model.inputs) == max_new_tokens


@pytest.mark.parametrize("context_length", [2, 3, 4])
def test_decoding_generation_context_length(context_length):
    model = _TransitionLM(context_length=context_length)
    prompt = torch.tensor([[0, 1], [1, 2]])
    expected = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])[:, : context_length + 1]

    actual = run_generate_naive_with_model(model, prompt, max_new_tokens=7)

    # A prefix at the context limit can predict one final token.
    torch.testing.assert_close(actual, expected)
    assert actual.shape[-1] == context_length + 1
    assert [tokens.shape[-1] for tokens in model.inputs] == list(range(prompt.shape[-1], context_length + 1))
    for tokens in model.inputs:
        torch.testing.assert_close(tokens, expected[:, : tokens.shape[-1]])


def test_decoding_generation_uses_last_position_and_full_prefix(decoding_model):
    prompt = torch.tensor([[0, 1], [1, 2]])
    original = prompt.clone()
    expected = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    actual = run_generate_naive_with_model(decoding_model, prompt, max_new_tokens=3)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(prompt, original)
    assert actual.dtype == prompt.dtype
    assert actual.device == prompt.device
    assert len(decoding_model.inputs) == 3
    for step, model_input in enumerate(decoding_model.inputs):
        torch.testing.assert_close(model_input, expected[:, : prompt.shape[-1] + step])


@pytest.mark.parametrize("prompt, expected", [([[3, 4]], [[3, 4, 5]]), ([[0, 1]], [[0, 1, 2, 3, 4, 5]])])
def test_decoding_generation_stops_at_eos(decoding_model, prompt, expected):
    prompt = torch.tensor(prompt)

    actual = run_generate_naive_with_model(decoding_model, prompt, max_new_tokens=8, eos_token_id=5)

    torch.testing.assert_close(actual, torch.tensor(expected))
    assert len(decoding_model.inputs) == len(expected[0]) - prompt.shape[-1]


@pytest.mark.parametrize("max_new_tokens", [3, 8])
def test_decoding_generation_batch_eos(decoding_model, max_new_tokens):
    prompt = torch.tensor([[3], [0]])
    expected = torch.tensor([[3, 4, 5, 5, 5, 5], [0, 1, 2, 3, 4, 5]])[:, : 1 + max_new_tokens]

    actual = run_generate_naive_with_model(decoding_model, prompt, max_new_tokens=max_new_tokens, eos_token_id=5)

    torch.testing.assert_close(actual, expected)
    assert len(decoding_model.inputs) == min(5, max_new_tokens)


def test_decoding_generation_eos_disabled(decoding_model):
    actual = run_generate_naive_with_model(decoding_model, torch.tensor([[3]]), max_new_tokens=7, eos_token_id=None)

    torch.testing.assert_close(actual, torch.tensor([[3, 4, 5, 0, 1, 2, 3, 4]]))
    assert len(decoding_model.inputs) == 7


def test_decoding_generation_eos_token_zero(decoding_model):
    actual = run_generate_naive_with_model(decoding_model, torch.tensor([[4]]), max_new_tokens=6, eos_token_id=0)

    torch.testing.assert_close(actual, torch.tensor([[4, 5, 0]]))
    assert len(decoding_model.inputs) == 2


def test_decoding_generation_forwards_sampling_options(decoding_model, monkeypatch):
    from cs336_basics import decoding

    calls = []

    def record_sample(logits, temperature=1.0, top_p=1.0):
        calls.append((logits.detach().clone(), temperature, top_p))
        return torch.ones((logits.shape[0], 1), dtype=torch.long, device=logits.device)

    monkeypatch.setattr(decoding, "sample_next_token", record_sample)
    actual = run_generate_naive_with_model(
        decoding_model, torch.tensor([[0, 1]]), max_new_tokens=2, temperature=0.7, top_p=0.8
    )

    torch.testing.assert_close(actual, torch.tensor([[0, 1, 1, 1]]))
    assert len(calls) == 2
    for logits, temperature, top_p in calls:
        torch.testing.assert_close(logits, decoding_model.logits[torch.tensor([1])])
        assert temperature == 0.7
        assert top_p == 0.8


@pytest.mark.parametrize("max_new_tokens", [0, 1, 3, 8])
def test_decoding_generation_from_weights(
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

    # A threshold below 1 / vocab_size retains only the most probable token.
    actual = run_generate_naive(
        **model_args,
        prompt_ids=in_indices,
        max_new_tokens=max_new_tokens,
        top_p=1e-6,
    )

    prompt_length = in_indices.shape[-1]
    expected_length = min(prompt_length + max_new_tokens, n_keys + 1)
    assert actual.shape == (in_indices.shape[0], expected_length)
    assert actual.dtype == in_indices.dtype
    assert actual.device == in_indices.device
    torch.testing.assert_close(actual[:, :prompt_length], original)
    torch.testing.assert_close(in_indices, original)
    with torch.no_grad():
        for position in range(prompt_length, expected_length):
            logits = run_transformer_lm(**model_args, in_indices=actual[:, :position])
            expected_token = logits[:, -1, :].argmax(dim=-1)
            torch.testing.assert_close(actual[:, position], expected_token)


def test_decoding_generation_disables_gradients(decoding_model):
    with torch.enable_grad():
        run_generate_naive_with_model(decoding_model, torch.tensor([[0, 1]]), max_new_tokens=2)

    assert decoding_model.grad_enabled == [False, False]
