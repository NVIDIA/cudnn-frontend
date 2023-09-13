import cudnn
import numpy as np
import cupy as cp
import sys
print("Example 2. Executing the Matmul + bias + relu graph")

if cudnn.backend_version() < 8500:
    print("cudnn version does not support matmul+bias fusion for specified layout")
    exit(0)

graph = cudnn.pygraph(io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

image = graph.tensor(name = "image", dim = [4,16,64], stride = [1024,1,16])
weight = graph.tensor(name = "weight", dim = [4,64,16], stride = [1024,1,64])
bias = graph.tensor(name = "bias", dim = [4,16,16],  stride = [256,1,16])

response = graph.matmul(name = "matmul", image = image, weight = weight)

output = graph.bias(name = "bias", input = response, bias = bias)

relu = graph.relu(name = "relu", input = output)
relu.set_output(True)

graph.check_support()

graph.build()

X_cpu = np.full([4,16,64], 1, dtype=np.half)
W_cpu = np.full([4,64,16], 1, dtype=np.half)
B_cpu = np.full([4,16,16], 2, dtype=np.half)

X_gpu = cp.asarray(X_cpu)
W_gpu = cp.asarray(W_cpu)
B_gpu = cp.asarray(B_cpu)
Y_gpu = cp.full([4,16,16], 0, dtype=cp.half)

workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
graph.execute({image : X_gpu, weight :  W_gpu, bias :  B_gpu, relu :  Y_gpu}, workspace)

Y_actual = cp.asnumpy(Y_gpu)

Y_expected = np.matmul(X_cpu, W_cpu) + B_cpu
Y_expected[Y_expected < 0] = 0

np.testing.assert_allclose(Y_actual, Y_expected)
