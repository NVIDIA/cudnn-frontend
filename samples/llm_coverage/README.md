# cuDNN Graph API Coverage on LLM Components

This directory contains test scripts to test the coverage of cuDNN Graph API on LLM components.

<table>
<tr>
    <th>Test Script</th>
    <th>Component</th>
    <th>Description</th>
</tr>
<tr>
    <td><a href="test_gqa_b+h+s+d.py">test_gqa_b+h+s+d.py</a></td>
    <td>GQA</td>
    <td>Group Query Attention with bfloat16/float16/float32 inputs and outputs. Input tensor is in BHSD dimension. Compute is in float32 data type. Forward and backward pass are tested.<br/><b>cuDNN Graph API is not supported for float32 I/O data type.</b></td>
</tr>
<tr>
    <td><a href="test_linear_1+bs+d.py">test_linear_1+bs+d.py</a></td>
    <td>Matmul</td>
    <td>Matmul with bfloat16/float16/float32 inputs and outputs, replaces the PyTorch nn.Linear module. Input tensor is in BSD dimension but reshaped to (1, BS, D) for cuDNN Graph API. Compute is in the same data type as I/O. Forward and backward pass are tested.</td>
</tr>
<tr>
    <td><a href="test_linear_b+s+d.py">test_linear_b+s+d.py</a></td>
    <td>Matmul</td>
    <td>Matmul with bfloat16/float16/float32 inputs and outputs, replaces the PyTorch nn.Linear module. Input tensor is in BSD dimension. Compute is in the same data type as I/O. <br/><b>cuDNN Graph API is not supported for bfloat16 data type.</b></td>
</tr>
<tr>
   <td><a href="test_linear_b+s+d_fp32compute.py">test_linear_b+s+d_fp32compute.py</a></td>
    <td>Matmul</td>
    <td>Matmul with bfloat16/float16/float32 inputs and outputs, replaces the PyTorch nn.Linear module. Input tensor is in BSD dimension. Compute is always in float32 data type. Forward and backward pass are tested.</td>
</tr>
<tr>
    <td><a href="test_linear+swish_1+bs+d.py">test_linear+swish_1+bs+d.py</a></td>
    <td>Matmul + Swish Fusion</td>
    <td>Matmul and pointwise swish with bfloat16/float16/float32 inputs and outputs, replaces the PyTorch nn.Linear module + F.silu() function. Input tensor is in BSD dimension but reshaped to (1, BS, D) for cuDNN Graph API. Compute is always in float32 data type. Only forward pass is tested. <br/><b>Backward pass is not supported by cuDNN Graph API yet.</b></td>
</tr>
<tr>
    <td><a href="test_swiglu_1+bs+d.py">test_swiglu_1+bs+d.py</a></td>
    <td>SwiGLU Fusion</td>
    <td>SwiGLU <code>(X @ W1) * swish(X @ W2)</code> with bfloat16/float16/float32 inputs and outputs, involves two matmul, one pointwise swish, and one pointwise multiplication. Input tensor is in BSD dimension but reshaped to (1, BS, D) for cuDNN Graph API. Compute is always in float32 data type. Only forward pass is tested. <br/><b>Backward pass is not supported by cuDNN Graph API yet.</b></td>
</tr>
<tr>
    <td><a href="test_swiglu_layer_1+bs+d.py">test_swiglu_layer_1+bs+d.py</a></td>
    <td>SwiGLU Fusion</td>
    <td>SwiGLU <code>((X @ W1) * swish(X @ W2)) @ W3</code> with bfloat16/float16/float32 inputs and outputs, involves three matmul, one pointwise swish, and one pointwise multiplication. Input tensor is in BSD dimension but reshaped to (1, BS, D). Compute is always in float32 data type. <br/><b>cuDNN Graph API is not supported for this case.</b></td>
</tr>
<tr>
    <td><a href="test_rmsnorm_b+s+d.py">test_rmsnorm_b+s+d.py</a></td>
    <td>RMSNorm</td>
    <td>RMSNorm with bfloat16/float16/float32 inputs and outputs. Input tensor is in BSD dimension. Compute is always in float32 data type. Forward and backward pass are tested.</td>
</tr>
<tr>
    <td><a href="test_rmsnorm_bs+d.py">test_rmsnorm_bs+d.py</a></td>
    <td>RMSNorm</td>
    <td>RMSNorm with bfloat16/float16/float32 inputs and outputs. Input tensor is in BSD dimension but reshaped to (BS, D) for cuDNN Graph API. Compute is always in float32 data type. Forward and backward pass are tested.</td>
</tr>
<tr>
    <td><a href="test_rope_b+s+h+d.py">test_rope_b+s+h+d.py</a></td>
    <td>RoPE</td>
    <td>RoPE with bfloat16/float16/float32 inputs and outputs. Pointwise multiplication and addition are applied to halves of the input tensor to produce the output. Input tensor is in BSHD dimension. Compute is always in float32 data type.<br/><b>cuDNN Graph API is not supported for this case.</b></td>
</tr>
</table>

To run these scripts, you can either run them with `python` command with cuDNN Frontend Python API installed (this repository):

```shell
python test_gqa_b+h+s+d.py
```

or run them with `pytest` command:

```shell
pytest test_gqa_b+h+s+d.py
```