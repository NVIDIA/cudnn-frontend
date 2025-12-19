"""Profile a concise PyTorch implementation of the Llama 3.1 8B model. The model
accepts the LlamaModel model weight from Hugging Face, and profiles the forward
and backward passes using NVTX.

To run this script with profiler:

nsys profile -f true --gpu-metrics-devices=cuda-visible --export=sqlite -o $OUTPUTFILE.nsys-rep \
    python $SCRIPT_PATH
"""

import dataclasses
import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, RMSNorm
from torch.nn.attention import SDPBackend, sdpa_kernel


# For type annotations
Tensor = torch.Tensor


@dataclasses.dataclass
class LlamaConfig:
    """Configuration class for LLaMA model hyperparameters.

    This matches the configuration of LLaMA 3.1 8B model with:
    - 32 transformer layers
    - 4096 hidden dimension
    - 32 attention heads
    - Grouped-query attention with 8 key-value heads
    """

    vocab_size: int = 128256  # Size of the tokenizer vocabulary
    max_position_embeddings: int = 131072  # Maximum sequence length
    hidden_size: int = 4096  # Dimension of hidden layers
    intermediate_size: int = 14336  # Dimension of MLP's hidden layer
    num_hidden_layers: int = 32  # Number of transformer layers
    num_attention_heads: int = 32  # Number of attention heads
    num_key_value_heads: int = 8  # Number of key-value heads for GQA
    rms_norm_eps: float = 1e-5  # Epsilon for RMSNorm


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input.

    This is a helper function for rotary position embeddings (RoPE).
    For a tensor of shape (..., d), it returns a tensor where the last
    d/2 dimensions are rotated by swapping and negating.

    Args:
        x: Input tensor of shape (..., d)

    Returns:
        Tensor of same shape with rotated last dimension
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)  # Concatenate with rotation


def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to a tensor.

    RoPE performs rotation in vector space based on position using
    trigonometric functions. This allows the model to learn relative
    positions in a more efficient way than absolute position embeddings.

    Args:
        x: Input tensor of shape (batch_size, seq_length, num_heads, head_dim)
        cos: Cosine position embeddings matching the shape of x
        sin: Sine position embeddings matching the shape of x
    """
    return (x * cos) + (rotate_half(x) * sin)


def get_inv_freq(N: float, dim: int) -> Tensor:
    """Get the inverse frequency for the RoPE with the Llama 3.1 scaling.
    Always computed in float32

    Args:
        N: Base, a large number
        dim: Size of hidden dimension, should be divisible by 2
    """
    N = float(N)
    dim = int(dim)
    # Llama 3.1 RoPE parameters
    factor = 8.0
    low_freq, high_freq = 1.0, 4.0
    context_len = 8192
    # Compute the inverse frequency based on the standard RoPE formula
    inv_freq = 1.0 / (N ** (torch.arange(0, dim, 2).float().to("cuda") / dim))
    # Compute the modified inverse frequency, then derive the smoothed inverse frequency
    wavelen = 2 * math.pi / inv_freq
    max_wavelen = context_len / low_freq
    min_wavelen = context_len / high_freq
    inv_freq = torch.where(wavelen > max_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (context_len / wavelen - low_freq) / (high_freq - low_freq)
    smoothed = (1 - smooth_factor) * inv_freq / factor + smooth_factor * inv_freq
    # Output inverse frequency as a mix of the two
    is_medium_freq = ~(wavelen < min_wavelen) * ~(wavelen > max_wavelen)
    inv_freq_final = torch.where(is_medium_freq, smoothed, inv_freq)
    return inv_freq_final


class RotaryPositionEncoding(nn.Module):
    """Rotary position encoding."""

    def __init__(self, dim: int, max_position_embeddings: int) -> None:
        """Initialize the RotaryPositionEncoding module

        Args:
            dim: The hidden dimension of the input tensor to which RoPE is applied
            max_position_embeddings: The maximum sequence length of the input tensor
        """
        super().__init__()
        # compute a matrix of n\theta_i
        N = 500_000.0
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.inv_freq = get_inv_freq(N, dim)
        position = torch.arange(max_position_embeddings).float().to("cuda")
        inv_freq = torch.cat((self.inv_freq, self.inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        # save cosine and sine matrices as buffers, not parameters
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def __repr__(self) -> str:
        return f"RotaryPositionEncoding(dim={self.dim}, max_position_embeddings={self.max_position_embeddings})"

    def forward(self, x: Tensor) -> Tensor:
        """Apply RoPE to tensor x

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_heads, head_dim)

        Returns:
            Output tensor of shape (batch_size, seq_length, num_heads, head_dim)
        """
        dtype = x.dtype
        seq_len = x.shape[1]
        # transform the cosine and sine matrices to 4D tensor and the same dtype as x
        cos = self.cos.to(dtype)[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin.to(dtype)[:seq_len].view(1, seq_len, 1, -1)
        # apply RoPE to x
        return apply_rotary_pos_emb(x, cos, sin)


class LlamaMLP(nn.Module):
    """MLP layer with SwiGLU activation.

    The architecture follows:
    1. Project input to intermediate size through two parallel layers
    2. Apply SwiGLU activation (multiply gate and up-projected inputs)
    3. Project back to hidden size
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        # Two parallel projections for SwiGLU
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        # Project back to hidden size
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu  # SwiGLU activation function

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU activation: multiply gate and up-projected inputs
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaAttention(nn.Module):
    """Multi-head attention with grouped-query attention and rotary embeddings.

    Grouped-query attention reduces computation by using fewer key-value heads
    than query heads, then sharing the same key-value heads across multiple queries.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads  # GQA: H_kv < H_q

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")

        # Linear layers for Q, K, V projections
        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        rope: Optional[RotaryPositionEncoding] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # Project inputs to Q, K, V
        torch.cuda.nvtx.range_push(":fwd.layer.attn.attn.qkv_proj")
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        torch.cuda.nvtx.range_pop()

        # Apply rotary position embeddings
        if rope is not None:
            torch.cuda.nvtx.range_push(":fwd.layer.attn.attn.rope")
            query_states = rope(query_states)
            key_states = rope(key_states)
            torch.cuda.nvtx.range_pop()

        # Transpose tensors from BSHD to BHSD dimension for scaled_dot_product_attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Use PyTorch's optimized attention implementation
        # setting is_causal=True is incompatible with setting explicit attention mask
        torch.cuda.nvtx.range_push(":fwd.layer.attn.attn.gqa")
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )
        torch.cuda.nvtx.range_pop()

        # Transpose output tensor from BHSD to BSHD dimension, reshape to 3D, and then project output
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        torch.cuda.nvtx.range_push(":fwd.layer.attn.attn.o_proj")
        attn_output = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop()
        return attn_output


class LlamaDecoderLayer(nn.Module):
    """Single transformer layer for LLaMA.

    Architecture:
    1. Input -> RMSNorm -> Self-Attention -> Residual
    2. RMSNorm -> MLP -> Residual
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        rope: Optional[RotaryPositionEncoding] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # First residual block: Self-attention
        with torch.cuda.nvtx.range(":fwd.layer.attn"):
            residual = hidden_states
            with torch.cuda.nvtx.range(":fwd.layer.attn.prenorm"):
                hidden_states = self.input_layernorm(hidden_states)
            with torch.cuda.nvtx.range(":fwd.layer.attn.attn"):
                attn_outputs = self.self_attn(hidden_states=hidden_states, rope=rope)
            hidden_states = attn_outputs + residual

        # Second residual block: MLP
        with torch.cuda.nvtx.range(":fwd.layer.mlp"):
            residual = hidden_states
            with torch.cuda.nvtx.range(":fwd.layer.mlp.prenorm"):
                hidden_states = self.post_attention_layernorm(hidden_states)
            with torch.cuda.nvtx.range(":fwd.layer.mlp.mlp"):
                hidden_states = self.mlp(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


class LlamaModel(nn.Module):
    """The full Llama model."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.rotary_emb = RotaryPositionEncoding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
        )

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Stack of transformer layers
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        output_hidden_states: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]]]:
        # Convert input token IDs to embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Initialize list to collect hidden states if requested
        all_hidden_states = () if output_hidden_states else None

        # Process through all transformer layers, accumulating hidden states as the input to each layer
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            with torch.cuda.nvtx.range(":fwd.layer"):
                hidden_states = layer(hidden_states, rope=self.rotary_emb)

        # Final layer norm, accumulate as the final hidden state
        with torch.cuda.nvtx.range(":fwd.output_norm"):
            hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return tuple of the output and the list of hidden states from all layers
        return [hidden_states, all_hidden_states]


# Create model with default config
test_config = LlamaConfig()
torch.set_default_device("cuda")
model = LlamaModel(test_config).to(torch.bfloat16)
print(time.time(), "model created")
state_dict = torch.load("llama3.1_8b_weights.bf16.pt", map_location="cuda")
print(time.time(), "state_dict loaded from disk")
model.load_state_dict(state_dict, strict=False)
print(time.time(), "model loaded")
del state_dict
print()

# Warm up once the forward pass and backward pass
BS, SEQ_LEN = 3, 10
x = torch.randint(0, test_config.vocab_size, (BS, SEQ_LEN), device="cuda", dtype=torch.int64)
y, hidden_states = model.forward(x, output_hidden_states=True)
target = torch.randn_like(y)
criterion = torch.nn.MSELoss()
loss = criterion(y, target)
loss.backward(retain_graph=True)  # warm up the cache
torch.cuda.synchronize()
print(time.time(), "Starting profiling")

# Profile the forward pass and backward pass
# SDPA backend: set to any of SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION, and SDPBackend.EFFICIENT_ATTENTION
with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]), torch.autograd.profiler.emit_nvtx():
    with torch.cuda.nvtx.range(":forward_backward"):
        with torch.cuda.nvtx.range(":fwd"):
            y, hidden_states = model.forward(x, output_hidden_states=True)
            torch.cuda.synchronize()
        loss = criterion(y, target)
        with torch.cuda.nvtx.range(":bwd"):
            loss.backward(retain_graph=True)  # warm up the cache
            torch.cuda.synchronize()

print(time.time(), "Profiling completed")
