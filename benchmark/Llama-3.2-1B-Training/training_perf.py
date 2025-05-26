"""Training Performance Measurement

This script is to measure the training performance of different backends in a Llama model.
Different batch sizes and sequence lengths are tested. Multiple training iterations are
run to collect timing data. Geometric mean of the timing is visualized afterwards.

This code uses models from Hugging Face Hub. You need to run with a valid token.
Consider to set the environment variables HF_TOKEN and HF_HOME appropriately.
Only the first GPU is used. You may set the environment variable CUDA_VISIBLE_DEVICES
before running this code to use a different GPU.

For more accurate results, it is recommended to lock the clock frequency of the GPU
using the following command:

    nvidia-smi -i 0 -lgc <min_clock>,<max_clock>
"""
import time

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# print system info
print(f"{torch.__version__ = }")
print(f"{torch.version.cuda = }")
print(f"{torch.cuda.is_available() = }")
print(f"{torch.cuda.device_count() = }")
print(f"{torch.cuda.current_device() = }")
print(f"{torch.cuda.get_device_name(torch.cuda.current_device()) = }")
print(f"{torch.backends.cudnn.version() = }")
print(f"{torch.backends.cudnn.enabled = }")

dtype = torch.bfloat16
device = torch.device("cuda:0")
torch.set_default_device(device)
torch.set_default_dtype(dtype)

model_name = "meta-llama/Llama-3.2-1B"
config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM(config).to(device).train()   # set norm layers to training mode
loss_fct = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

# Configuration matrix to test
batch_seqlen = [(24, 768), (12, 1024), (6, 2048), (3, 4096), (2, 8192), (1, 16384)]
backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]

# Run timing experiments
warmup_iterations = 5  # num of training iterations to run for warmup
measure_iterations = 100  # num of training iterations to run to measure for timing
data = []
for batch_size, seq_len in batch_seqlen:
    assert seq_len < tokenizer.model_max_length, "seqlen must be less than the model max length"
    # create random tensors
    #  - input embedding tensor to simulate a batch of input token sequences converted into embeddings
    #  - attention mask of all ones for full attention
    #  - random target to compute cross entropy loss in training loop
    shape = (batch_size, seq_len, config.hidden_size)
    inputs_embeds = torch.randn(*shape, dtype=dtype, device=device)
    attention_mask = torch.ones(*shape[:2], dtype=torch.int64, device=device)
    target = torch.randint(2, config.vocab_size-2, shape[:2], dtype=torch.int64, device=device)
    for backend in backends:
        backend_name = str(backend).split(".")[-1]
        print(f"Timing {backend_name} with batch_size={batch_size} and seq_len={seq_len}")
        with sdpa_kernel(backends=[backend]):
            # warmup iterations: to minimize the effect of system cache
            for _ in range(warmup_iterations):
                output = model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                loss = loss_fct(output.logits.view(-1, config.vocab_size), target.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            start = time.time()
            # measure iterations: per-iteration time obtained by averaging
            for _ in range(measure_iterations):
                output = model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                loss = loss_fct(output.logits.view(-1, config.vocab_size), target.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()   # wait for all kernels to finish for accurate timing
            duration = time.time() - start
            data.append((backend_name, batch_size, seq_len, duration/measure_iterations))

# Process stats
df = pd.DataFrame(data, columns=["backend", "batch_size", "seq_len", "time"])
df["label"] = "BS=" + df["batch_size"].astype(str) + " SL=" + df["seq_len"].astype(str)
# compute the speedup w.r.t. CUDNN_ATTENTION
df["speedup_label"] = df["backend"] + " vs EFFICIENT_ATTENTION"
df["speedup"] = df.apply(
    lambda row: df.loc[(df["backend"] == "EFFICIENT_ATTENTION") & (df["batch_size"] == row["batch_size"]) & (df["seq_len"] == row["seq_len"]), "time"].values[0] / row["time"],
    axis=1)
df.to_csv("training_timing.csv", index=False)

# Create plots
label_order = [f"BS={b} SL={s}" for b, s in batch_seqlen]  # x-axis order
hue_order = ["CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"]
g = sns.barplot(data=df, x="label", y="time", hue="backend",
                palette=["#76B900", "orchid", "royalblue"], order=label_order, hue_order=hue_order)
g.set_title("\nTraining Iteration Time")
g.set(xlabel="Batch size and sequence length", ylabel="Mean iteration time (s), lower is better")
g.get_legend().set_title("")
plt.legend(fontsize=8)
plt.xticks(rotation=10, size=8)
plt.tight_layout()
plt.savefig("iteration_time.png", dpi=300)

plt.clf()
hue_order = ["CUDNN_ATTENTION vs EFFICIENT_ATTENTION", "FLASH_ATTENTION vs EFFICIENT_ATTENTION"]
g = sns.barplot(data=df[df["speedup_label"]!="EFFICIENT_ATTENTION vs EFFICIENT_ATTENTION"],
                x="label", y="speedup", hue="speedup_label",
                palette=["#76B900", "orchid"], order=label_order, hue_order=hue_order)
for container in g.containers:
    g.bar_label(container, fmt="%.2f", fontsize=6)
g.set_title("Per-iteration Speed-up of\ncuDNN/Flash Attention Backend vs Efficient Attention")
g.set(xlabel="Batch size and sequence length", ylabel="Speed-up ratio, higher is better")
g.get_legend().set_title("")
plt.legend(fontsize=8)
plt.xticks(rotation=10, size=8)
plt.tight_layout()
plt.savefig("speedup.png", dpi=300)
