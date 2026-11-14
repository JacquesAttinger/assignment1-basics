import pytest
import torch
import torch.nn.functional as F

from cs336_basics.nn import MultiHeadSelfAttention, scaled_dot_product_attention


@pytest.mark.parametrize("mask_kind", ["none", "broadcast", "fully_masked"])
def test_attention_matches_pytorch_outputs_and_gradients(mask_kind):
    generator = torch.Generator().manual_seed(42)
    q, k, v = [
        torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
        for shape in [(2, 3, 4, 5), (2, 3, 6, 5), (2, 3, 6, 7)]
    ]
    mask = None
    if mask_kind != "none":
        mask = torch.ones(4, 6, dtype=torch.bool).tril()
        if mask_kind == "fully_masked":
            mask[0] = False

    actual = scaled_dot_product_attention(q, k, v, mask)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, expected)

    upstream = torch.randn(actual.shape, generator=generator, dtype=actual.dtype)
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("num_heads", [1, 3])
def test_multihead_matches_independent_heads_and_gradients(batch_shape, num_heads):
    torch.manual_seed(17)
    attention = MultiHeadSelfAttention(12, num_heads, dtype=torch.float64)
    x = torch.randn(*batch_shape, 5, 12, dtype=torch.float64, requires_grad=True)
    heads = []
    for q_weight, k_weight, v_weight in zip(
        attention.q_proj_weight.chunk(num_heads, dim=0),
        attention.k_proj_weight.chunk(num_heads, dim=0),
        attention.v_proj_weight.chunk(num_heads, dim=0),
    ):
        heads.append(F.scaled_dot_product_attention(
            F.linear(x, q_weight), F.linear(x, k_weight), F.linear(x, v_weight), is_causal=True
        ))
    expected = F.linear(torch.cat(heads, dim=-1), attention.o_proj_weight)
    actual = attention(x)
    torch.testing.assert_close(actual, expected)

    upstream = torch.randn_like(actual)
    inputs = (x, *attention.parameters())
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, inputs, upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_multihead_cannot_attend_to_future_tokens():
    torch.manual_seed(21)
    attention = MultiHeadSelfAttention(12, 3)
    x = torch.randn(2, 5, 12)
    changed = x.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10
    with torch.no_grad():
        original_output = attention(x)
        changed_output = attention(changed)
        prefix_output = attention(x[:, :3])
        single_output = attention(x[:, :1])
    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])
    torch.testing.assert_close(original_output[:, :3], prefix_output)
    torch.testing.assert_close(original_output[:, :1], single_output)


@pytest.mark.parametrize("d_model,num_heads", [(12, 0), (0, 3), (12, -3), (12, 5)])
def test_multihead_rejects_invalid_head_dimensions(d_model, num_heads):
    with pytest.raises(ValueError, match="num_heads"):
        MultiHeadSelfAttention(d_model, num_heads)
