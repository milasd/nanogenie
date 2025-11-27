import pytest
import torch

from nanogenie.models import STTransformerBlock


B, T, H, W, d_model = 1, 2, 2, 2, 8


@pytest.fixture
def st_block_config():
    """Config w/ small dims for ST Transformer Block creation."""
    return {"d_model": d_model, "num_heads": 2, "ffn_dim": 16}


@pytest.fixture
def st_block(st_block_config):
    """Create the ST Transformer block to be used in all tests."""
    return STTransformerBlock(**st_block_config)


@pytest.fixture
def test_input():
    """Create a random input [B, T, H, W, d_model] tensor for testing."""
    return torch.randn(B, T, H, W, d_model)


@pytest.fixture
def causal_mask(test_input):
    """
    Causal mask for temporal attention testing.

    Shape: (T, T)
    mask[i,j] = True means position i CANNOT attend to position j

    Eg. (T=4):
        [[False,  True,  True,  True],   # Frame 0 can't see future
         [False, False,  True,  True],   # Frame 1 can't see 2,3
         [False, False, False,  True],   # Frame 2 can't see 3
         [False, False, False, False]]   # Frame 3 can see all past
    """
    T_t = test_input.shape[1]
    mask = torch.triu(torch.ones(T_t, T_t), diagonal=1).bool()
    return mask


def test_spatial_attention_shape(st_block, test_input):
    # assert that spatial attention final output has correct shape
    # [B, T, H, W, d_model]
    x_spatial = st_block.spatial_attention(test_input)
    assert x_spatial.shape == (B, T, H, W, d_model)


def test_temporal_attention_shape(st_block, causal_mask, test_input):
    x_temporal = st_block.temporal_attention(x=test_input, causal_mask=causal_mask)
    assert x_temporal.shape == (B, T, H, W, d_model)


def test_ffn_output_shape(st_block, test_input):
    x_ffn = st_block.ffn_residual(test_input)
    assert x_ffn.shape == (B, T, H, W, d_model)


def test_forward_shape(st_block, causal_mask, test_input):
    x_out = st_block(x=test_input, causal_mask=causal_mask)
    assert x_out.shape == (B, T, H, W, d_model)


def test_spatial_attention_intermediate_reshape(st_block, test_input):
    """Test if spatial attention correctly reshapes input to [B*T, H*W, D]"""
    captured = {}

    def hook(module, input, output):
        captured["spatial_attn_reshape"] = input[0].shape

    # Register hook on the spatial attention layer
    handle = st_block.spatial_attn.register_forward_hook(hook)
    st_block.spatial_attention(test_input)
    handle.remove()

    assert captured["spatial_attn_reshape"] == (B * T, H * W, d_model)


def test_temporal_attention_intermediate_reshape(st_block, causal_mask, test_input):
    """Test if temporal attention correctly reshapes input to [B*H*W, T, D]"""
    captured = {}

    def hook(module, input, output):
        captured["temporal_attn_reshape"] = input[0].shape

    handle = st_block.temporal_attn.register_forward_hook(hook)
    st_block.temporal_attention(x=test_input, causal_mask=causal_mask)
    handle.remove()

    assert captured["temporal_attn_reshape"] == (B * H * W, T, d_model)


def test_causal_mask_prevents_future_attention(st_block, test_input, causal_mask):
    """Test if temporal attention with mask differs from without mask"""
    x_with_mask = st_block.temporal_attention(test_input, causal_mask)
    x_without_mask = st_block.temporal_attention(test_input, None)

    assert not torch.allclose(x_with_mask, x_without_mask)


def test_single_frame(st_block_config):
    """Test with T=1 (single frame)"""
    st_transf = STTransformerBlock(**st_block_config)
    x = torch.randn(B, 1, H, W, d_model)
    mask = None  # No mask needed for single frame!
    output = st_transf(x, mask)

    assert output.shape == (B, 1, H, W, d_model)


def test_single_spatial_location(st_block_config, causal_mask):
    """Test with H=1, W=1 (single spatial location)"""
    st_transf = STTransformerBlock(**st_block_config)
    x = torch.randn(B, T, 1, 1, d_model)
    output = st_transf(x, causal_mask)

    assert output.shape == (B, T, 1, 1, d_model)


def test_deterministic_output(st_block, causal_mask, test_input):
    """Test if same input will produce same output"""
    torch.manual_seed(22)
    out1 = st_block(test_input, causal_mask)
    torch.manual_seed(22)
    out2 = st_block(test_input, causal_mask)

    assert torch.allclose(out1, out2)
