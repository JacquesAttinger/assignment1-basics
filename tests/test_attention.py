import pytest
import torch
import torch.nn.functional as F

from cs336_basics.nn import scaled_dot_product_attention


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
