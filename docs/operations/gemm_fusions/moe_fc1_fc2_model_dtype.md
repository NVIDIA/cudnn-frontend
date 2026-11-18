# MoE FC1 / FC2 dtype

## MoE dtype

| Model | FC1 Weight | FC1 Input | FC2 Weight | FC2 Input |
|---|---|---|---|---|
| DeepSeek V4.1 Flash / 1017 | MXFP4 | MXFP8 | MXFP4 | MXFP8 |
| DeepSeek V4 Flash / DSpark / Vision | MXFP4 | FP8 (1×128, UE8M0) | MXFP4 | FP8 (1×128, UE8M0) |
| DeepSeek V4 Flash-Base | FP8 (128×128) | FP8 (1×128) | FP8 (128×128) | FP8 (1×128) |
| DeepSeek V4 Pro / DSpark / 0813 | MXFP4 | FP8 (1×128, UE8M0) | MXFP4 | FP8 (1×128, UE8M0) |
| DeepSeek V4 Pro-Base | FP8 (128×128) | FP8 (1×128) | FP8 (128×128) | FP8 (1×128) |
| DeepSeek V3 / V3-0324 / R1 / R1-0528 / Prover-V2-671B | FP8 (128×128) | FP8 (1×128) | FP8 (128×128) | FP8 (1×128) |
| DeepSeek V3.1 / V3.2 / Math-V2 | FP8 (128×128, UE8M0) | FP8 (1×128, UE8M0) | FP8 (128×128, UE8M0) | FP8 (1×128, UE8M0) |
| DeepSeek VL2 | BF16 | BF16 | BF16 | BF16 |
| DeepSeek MoE-16B / VL2-Small; Kimi Moonlight | BF16 | BF16 | BF16 | BF16 |
| DeepSeek V2-Lite / Coder-V2-Lite; Kimi-VL | BF16 | BF16 | BF16 | BF16 |
| DeepSeek VL2-Tiny | BF16 | BF16 | BF16 | BF16 |
| DeepSeek V2 / V2.5 / Coder-V2 | BF16 | BF16 | BF16 | BF16 |
| Qwen3.8-2.4T-A95B | BF16 | BF16 | BF16 | BF16 |
| Qwen3.8-2.4T-A95B-FP8 | FP8 (128×128) | FP8 (1×128) | FP8 (128×128) | FP8 (1×128) |
| Qwen3.5 / Qwen3.6 / AgentWorld-35B-A3B | BF16 | BF16 | BF16 | BF16 |
| Qwen3.5-122B-A10B | BF16 | BF16 | BF16 | BF16 |
| Qwen3.5-397B-A17B | BF16 | BF16 | BF16 | BF16 |
| Qwen3.5-397B-A17B-FP8 | FP8 (128×128) | FP8 (1×128) | FP8 (128×128) | FP8 (1×128) |
| Qwen3 / Coder / VL-30B-A3B; Omni thinker | BF16 | BF16 | BF16 | BF16 |
| Qwen3 / VL-235B-A22B | BF16 | BF16 | BF16 | BF16 |
| Qwen3-Next-80B-A3B / Coder-Next | BF16 | BF16 | BF16 | BF16 |
| Qwen3-Omni talker | BF16 | BF16 | BF16 | BF16 |
| Qwen3-Coder-480B-A35B | BF16 | BF16 | BF16 | BF16 |
| Qwen2-57B-A14B | BF16 | BF16 | BF16 | BF16 |
| Qwen1.5-MoE-A2.7B | BF16 | BF16 | BF16 | BF16 |
| Mistral Small 4 | FP8 (per-tensor) | FP8 (per-tensor, static) | FP8 (per-tensor) | FP8 (per-tensor, static) |
| Leanstral / Leanstral 1.5 | FP8 | ? | FP8 | ? |
| Mixtral-8x22B | BF16 | BF16 | BF16 | BF16 |
| Mixtral-8x7B | BF16 | BF16 | BF16 | BF16 |
| Kimi K3 | MXFP4 | MXFP8 | MXFP4 | MXFP8 |
| Kimi K2 (Base / Instruct / 0905) | FP8 (128×128) | FP8 (1×128) | FP8 (128×128) | FP8 (1×128) |
| Kimi K2-Thinking / K2.5 / K2.6 / K2.7-Code | INT4 (g32) | BF16 | INT4 (g32) | BF16 |
| Kimi Linear-48B-A3B | BF16 | BF16 | BF16 | BF16 |
| GLM-5.3-Flash | FP8 (128×128) | FP8 (1×128) | FP8 (128×128) | FP8 (1×128) |
| GLM-5 / 5.1 / 5.2 / 5.3 | BF16 | BF16 | BF16 | BF16 |
| GLM-5 / 5.1 / 5.2 / 5.3 -FP8 | FP8 (128×128) | FP8 (1×128) | FP8 (128×128) | FP8 (1×128) |
| GLM-4.7-Flash | BF16 | BF16 | BF16 | BF16 |
| GLM-4.5 / 4.6 / 4.7 | BF16 | BF16 | BF16 | BF16 |
| GLM-4.5-Air / 4.5V / 4.6V | BF16 | BF16 | BF16 | BF16 |
| MiniMax M3 | BF16 | BF16 | BF16 | BF16 |
| MiniMax M3-MXFP8 | MXFP8 | MXFP8 | MXFP8 | MXFP8 |
| MiniMax M3.1-Flash | NVFP4 | NVFP4 | NVFP4 | NVFP4 |
| MiniMax M2 / M2.1 / M2.5 / M2.7 | FP8 (128×128) | FP8 (1×128) | FP8 (128×128) | FP8 (1×128) |
| MiniMax Text-01 / VL-01 / M1 | BF16 | BF16 | BF16 | BF16 |
| gpt-oss / safeguard-20b | MXFP4 | BF16 | MXFP4 | BF16 |
| gpt-oss / safeguard-120b | MXFP4 | BF16 | MXFP4 | BF16 |
| Qwen3.8-Flash-Next | BF16 | BF16 | BF16 | BF16 |
| Mistral Large 3 | FP8 (128×128) | FP8 (1×128) | FP8 (128×128) | FP8 (1×128) |
| Llama 4 Scout | BF16 | BF16 | BF16 | BF16 |
| Llama 4 Maverick | BF16 | BF16 | BF16 | BF16 |
| Llama 4 Maverick-FP8 | FP8 (per-channel) | FP8 (per-token) | FP8 (per-channel) | FP8 (per-token) |
| DeepSeek V4.1 DSpark draft | MXFP4 | MXFP8 | MXFP4 | MXFP8 |


| Model | Link |
|---|---|
| DeepSeek V4.1 Flash / 1017 | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference |
| DeepSeek V4 Flash / DSpark / Vision | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference |
| DeepSeek V4 Flash-Base | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/blob/main/config.json |
| DeepSeek V4 Pro / DSpark / 0813 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json |
| DeepSeek V4 Pro-Base | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base/blob/main/config.json |
| DeepSeek V3 / V3-0324 / R1 / R1-0528 / Prover-V2-671B | https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/ec67248f61244f350422cdfd2e78258af58edff9/README_WEIGHTS.md |
| DeepSeek V3.1 / V3.2 / Math-V2 | https://friendli.ai/models/deepseek-ai/DeepSeek-V3.1 |
| DeepSeek VL2 | https://huggingface.co/deepseek-ai/deepseek-vl2/blame/6126a594b0fa8cb48edccbca371cb4705dac4228/config.json |
| DeepSeek MoE-16B / VL2-Small; Kimi Moonlight | https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blame/main/config.json |
| DeepSeek VL2-Tiny | https://huggingface.co/deepseek-ai/deepseek-vl2-tiny/blame/58925b0e404c7aabcbd483419e5b8ca572a00119/config.json |
| Qwen3.8-2.4T-A95B-FP8 | https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-qwen3.8-qwen3.5-on-trtllm.html |
| Qwen3.5 / Qwen3.6 / AgentWorld-35B-A3B | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5 |
| Qwen3.5-122B-A10B | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5 |
| Qwen3.5-397B-A17B | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5 |
| Qwen3.5-397B-A17B-FP8 | https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8 |
| Mistral Small 4 | https://huggingface.co/mistralai/Mistral-Small-4-119B-2603/blob/main/config.json |
| Kimi K3 | https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json |
| Kimi K2 (Base / Instruct / 0905) | https://huggingface.co/moonshotai/Kimi-K2-Instruct |
| Kimi K2-Thinking / K2.5 / K2.6 / K2.7-Code | https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/main/config.json |
| GLM-5.3-Flash | https://huggingface.co/8times4/apex-flash-1-FP8 |
| GLM-5 / 5.1 / 5.2 / 5.3 | https://recipes.vllm.ai/zai-org/GLM-5.1 |
| GLM-5 / 5.1 / 5.2 / 5.3 -FP8 | https://huggingface.co/Blackfrost-Research/GLM-5.2-DERISKED-FP8 |
| MiniMax M3 | https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-minimax-m3-on-amd-instinct-gpus.html |
| MiniMax M3-MXFP8 | https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8/commit/2a60e16e52376b8eb7a7a94c930401c1fe12b709 |
| MiniMax M3.1-Flash | https://vllm.ai/blog/2026-10-23-minimax-m31-flash |
| MiniMax M2 / M2.1 / M2.5 / M2.7 | https://huggingface.co/MiniMaxAI/MiniMax-M2.1/commit/500a52ef4b04096bbc9b02b41c54d08dbcc9480a |
| gpt-oss / safeguard-20b | https://huggingface.co/blog/faster-transformers |
| gpt-oss / safeguard-120b | https://huggingface.co/blog/faster-transformers |
| Mistral Large 3 | https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/blob/main/params.json |
| Llama 4 Scout | https://developers.redhat.com/articles/2025/04/05/llama-4-herd-here-day-zero-inference-support-vllm |
| Llama 4 Maverick | https://developers.redhat.com/articles/2025/04/05/llama-4-herd-here-day-zero-inference-support-vllm |
| Llama 4 Maverick-FP8 | https://developers.redhat.com/articles/2025/04/05/llama-4-herd-here-day-zero-inference-support-vllm |
| DeepSeek V4.1 DSpark draft | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference |

