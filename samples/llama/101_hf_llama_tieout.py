"""This script creates a PyTorch implementation of the Llama 3.1 8B model
using Hugging Face transformers library and loads the pretrained weights.

Then randomize an input tensor of shape (3,10) of integer token IDs, processed
through the model, and perform a backward pass.

This script saves into "tensors-bf16-tieout.pt" the:
- integer input tensor
- RoPE sine and cosine values
- inverse frequency values used in RoPE
- model output tensor (final hidden state)
- list of the hidden states to all transformer layers
- a random target tensor used to compute the MSE loss
- the gradient of the entry embed layer
- the gradient of the output norm layer
"""

import torch
from transformers import LlamaModel, AutoConfig

# Load the model with the downloaded weights in bfloat16 precision
torch.set_default_device("cuda")
REPO_ID = "meta-llama/Llama-3.1-8B"
config = AutoConfig.from_pretrained(REPO_ID)
print(config)
model = LlamaModel(config).to(torch.bfloat16)
model.load_state_dict(torch.load("llama3.1_8b_weights.bf16.pt", map_location="cuda"), strict=False)
print(model)

# Run a forward pass
BS, SEQ_LEN = 3, 10
x = torch.randint(0, model.config.vocab_size, (BS, SEQ_LEN))
y = model.forward(x, output_hidden_states=True, output_attentions=False)
print("Output shape:", y[0].shape, y[0].dtype)  # Print the shape of hidden states
assert y[0].dtype == torch.bfloat16, "Output should be in bfloat16 precision"
assert len(y[-1]) == 33, "Llama 8B with 32 layers should have 33 hidden states"

# Extract RoPE sine and cosine values
x_embed = model.embed_tokens(x)
position_ids = torch.arange(SEQ_LEN, device=x.device).unsqueeze(0).to(torch.float32)
x_rope = model.rotary_emb(x_embed, position_ids)  # get sine and cosine values
inv_freq = model.rotary_emb.inv_freq
assert model.rotary_emb.attention_scaling == 1.0, "Attention scaling in Llama 8B should be 1.0"

# Run a backward pass
# MSE as loss function to take every element into account
target = torch.randn_like(y[0])  # random target tensor
criterion = torch.nn.MSELoss()
loss = criterion(y[0], target)
loss.backward()
print(
    "Gradient shape:",
    model.embed_tokens.weight.grad.shape,
    model.embed_tokens.weight.grad.dtype,
)
grad_embed = model.embed_tokens.weight.grad  # grad on entry embed layer
grad_norm = model.norm.weight.grad  # grad on output norm layer

# save tensors to file
tensors = [x, x_rope, inv_freq, y[0], y[-1], target, grad_embed, grad_norm]
torch.save(tensors, "tensors-bf16-tieout.pt")
