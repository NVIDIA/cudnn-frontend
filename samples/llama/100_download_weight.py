"""This script downloads the Llama 3.1 8B model weights from Hugging Face and saves them in a PyTorch .pt file.
The weights are saved in a PyTorch .pt file named "llama3.1_8b_weights.bf16.pt".
"""

import os

if "HF_HOME" not in os.environ:
    print("HF_HOME not set. Default may be ~/.cache/huggingface")
if "HF_TOKEN" not in os.environ and "HF_TOKEN_PATH" not in os.environ:
    print("HF_TOKEN and HF_TOKEN_PATH not set. You may not be able to download the weights.")

import torch
from transformers import LlamaModel

# Load the model with the downloaded weights in bfloat16 precision
REPO_ID = "meta-llama/Llama-3.1-8B"
model = LlamaModel.from_pretrained(REPO_ID, torch_dtype=torch.bfloat16)

# Save the model weights in a PyTorch .pt file
torch.save(model.state_dict(), "llama3.1_8b_weights.bf16.pt")
print("Model weights saved")
