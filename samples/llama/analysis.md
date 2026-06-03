# Analyzing Performance Differences Between PyTorch and cuDNN in Llama 3.1 8B Model

On this page, we briefly illustrate the motivation of replacing several PyTorch modules with cuDNN and demonstrate how to measure the performance hotspots in the model and analyze the benefits of using cuDNN.

A more thorough analysis of the training performance differences between PyTorch and cuDNN in Llama model is at [the benchmark directory](../../benchmark/Llama-3.2-1B-Training/).

## Workflow

To reproduce the results in this page, you should follow the following steps:

1. Download the Llama 3.1 8B model weights from Hugging Face and save them in the current directory as `llama3.1_8b_weights.bf16.pt`, using the following command:

    python 100_download_weight.py

2. Run the script `104_torch_llama_nvtx.py` and  `105_cudnn_llama_nvtx.py` to profile the GPU performance of the PyTorch and cuDNN implementations. They will store the performance data in the SQLite files `104_torch_llama_nvtx.sqlite` and `105_cudnn_llama_nvtx.sqlite` respectively. You should run the following commands:

    nsys profile -f true --gpu-metrics-devices=cuda-visible --export=sqlite -o 104_torch_llama_nvtx.nsys-rep \
        python 104_torch_llama_nvtx.py

    nsys profile -f true --gpu-metrics-devices=cuda-visible --export=sqlite -o 105_cudnn_llama_nvtx.nsys-rep \
        python 105_cudnn_llama_nvtx.py

3. Analyze the performance data using the script `decode_nvtx_profile.py`. The result will be printed to the console. You should run the following commands:

    python decode_nvtx_profile.py 104_torch_llama_nvtx.sqlite
    python decode_nvtx_profile.py 105_cudnn_llama_nvtx.sqlite


## Results

Below is the output of the final step of the aforementioned workflow on a NVIDIA B100 GPU:

```
Analyzing CUDA profile from: 104_torch_llama_nvtx.sqlite
Found 8011 kernels, 14025 CPU calls, and 20796 events
nvtx event                       num calls    kernel count    kernel time total
-----------------------------  -----------  --------------  -------------------
:fwd                                     1           1,257            6,968,861
:fwd.layer                              32           1,248            6,936,989
:fwd.layer.attn                         32             768            3,507,999
:fwd.layer.attn.attn                    32             480            2,477,727
:fwd.layer.attn.attn.gqa                32              32              313,248
:fwd.layer.attn.attn.o_proj             32              32              345,280
:fwd.layer.attn.attn.qkv_proj           32              96              864,896
:fwd.layer.attn.attn.rope               32             320              954,303
:fwd.layer.attn.prenorm                 32             256              965,216
:fwd.layer.mlp                          32             480            3,428,990
:fwd.layer.mlp.mlp                      32             192            2,401,055
:fwd.layer.mlp.prenorm                  32             256              959,711
:fwd.output_norm                         1               8               28,896

Analyzing CUDA profile from: 105_cudnn_llama_nvtx.sqlite
Found 5475 kernels, 14155 CPU calls, and 15246 events
nvtx event                       num calls    kernel count    kernel time total
-----------------------------  -----------  --------------  -------------------
:fwd                                     1             802            4,936,635
:fwd.layer                              32             800            4,931,195
:fwd.layer.attn                         32             544            2,426,428
:fwd.layer.attn.attn                    32             480            2,279,869
:fwd.layer.attn.attn.gqa                32              32              144,128
:fwd.layer.attn.attn.o_proj             32              32              336,415
:fwd.layer.attn.attn.qkv_proj           32              96              844,831
:fwd.layer.attn.attn.rope               32             320              954,495
:fwd.layer.attn.prenorm                 32              32               83,231
:fwd.layer.mlp                          32             256            2,504,767
:fwd.layer.mlp.mlp                      32             192            2,353,343
:fwd.layer.mlp.prenorm                  32              32               84,160
:fwd.output_norm                         1               1                2,528
```

Rearranging, the following table summarizes the performance data:

| nvtx event | num calls | pytorch kernel count | pytorch kernel time total | cudnn kernel count | cudnn kernel time total | cudnn speedup |
|------------|-----------|--------------|-------------------|--------------|-------------------|---------------|
| :fwd | 1 | 1257 | 6,968,861.00 | 802 | 4,936,635.00 | 41.2% |
| :fwd.layer | 32 | 1248 | 6,936,989.00 | 800 | 4,931,195.00 | 40.7% |
| :fwd.layer.attn | 32 | 768 | 3,507,999.00 | 544 | 2,426,428.00 | 44.6% |
| :fwd.layer.attn.attn | 32 | 480 | 2,477,727.00 | 480 | 2,279,869.00 | 8.7% |
| :fwd.layer.attn.attn.gqa | 32 | 32 | 313,248.00 | 32 | 144,128.00 | 117.3% |
| :fwd.layer.attn.attn.o_proj | 32 | 32 | 345,280.00 | 32 | 336,415.00 | 2.6% |
| :fwd.layer.attn.attn.qkv_proj | 32 | 96 | 864,896.00 | 96 | 844,831.00 | 2.4% |
| :fwd.layer.attn.attn.rope | 32 | 320 | 954,303.00 | 320 | 954,495.00 | 0.0% |
| :fwd.layer.attn.prenorm | 32 | 256 | 965,216.00 | 32 | 83,231.00 | 1059.7% |
| :fwd.layer.mlp | 32 | 480 | 3,428,990.00 | 256 | 2,504,767.00 | 36.9% |
| :fwd.layer.mlp.mlp | 32 | 192 | 2,401,055.00 | 192 | 2,353,343.00 | 2.0% |
| :fwd.layer.mlp.prenorm | 32 | 256 | 959,711.00 | 32 | 84,160.00 | 1040.3% |
| :fwd.output_norm | 1 | 8 | 28,896.00 | 1 | 2,528.00 | 1043.0% |

The table shows the number of time a NVTX event is encountered, the number of CUDA kernels involved in total, and the total time (nanoseconds) spent in these CUDA kernels. The cuDNN speedup is calculated as the ratio of the PyTorch kernel time total to the cuDNN kernel time total.

From there, we can see that the cuDNN implementation is generally faster than the PyTorch implementation, especially the normalization layers. In fact switching the normalization implementation to cuDNN is the major contributor to make the forward pass 41% faster. Comparatively, our method of replacing the linear layers does not provide any material advantage. This is an example of how you can experiment in a larger model to find the benefits of using cuDNN.