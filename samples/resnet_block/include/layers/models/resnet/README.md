# Macro blocks to create a Residual Network

## What this directory contains:
* Header files that implement macro mixed-precision blocks to create a Residual network
* Param builders for each of the blocks that you, you, need to configure for the blocks. The necessary params you need to configre will be documented below
* "Device pointer stores", objects that contain GPU device pointers needed for each of the blocks. Again, you are responsible for setting the necessary device pointers. The necessary device pointers will be documented below.

## Guide to using these blocks

To use any of these blocks, first, ensure `cudnn_frontend.h` has been included in your configuration. Then, you'll have access to the necessary blocks and builders. 

In the [original ResNet Paper](https://arxiv.org/abs/1512.03385?context=cs), a ResNet contains a stem block, a stack of building/bottleneck blocks, and a classifier block. A stem block contains a 7x7 convolution on the input, followed by a batch normalization, and followed by a maxpooling layer. Each bottleneck block contains 3, 3x3 convolutions followed by a batch normalization and a ReLu. There is also the residual path, with one flavor of the network containing a 1x1 convolution and the second being a direct connection from input to output. Finally, the classifier block is a fully connected layer that outputs probabilities for classes.

This repo can be used to easilly build mixed-precision ResNets and networks using ResNet backbones by stacking these blocks with our API
* Residual forward block, found in `cudnn_frontend_residual_forward_block.h`

Note that we've separated the forward pass and backward for encapsulation and readibilty. However, both types of blocks communicate to each other via a shared `Block Params` object and a `Block Device Pointer Store` object. These objects house the parameters for the block and the GPU device pointers for outputs. Backward blocks will be part of future cudnn_frontend release.

## Parent block: `IBlock`
All of these blocks inherit from a parent block called `IBlock` found in `layers/common/include/cudnn_frontend_layer_interface.h`. In the future, we may want to create different types of blocks for different networks, so a parent block is created. `IBlock` contains methods such as `buildOperationGraph()`, `execute()`, etc. for children blocks to implement. We handle operation graph building and execution internally, so you as the user will not have to worry about that.

## IMPORTANT
You should look at the samples in `resnet_sample.cpp` alongside `resnet_test_list.cpp` to see sample usage of these blocks!

## Block params
Each block has a corresponding "params builder" that sets up the necessary parameters for each of the block in the form of a `params` object. Parameters include filter sizes, input sizes, output sizes, etc. Each builder, after finalizing, will return a `cudnnStatus_t` denoting whether or not the builder could build a `params` object. you shuold check the status to ensure it's a `CUDNN_STATUS_SUCCESS`. Let's take a look at an example of configuring a FP16 Residual block (bottleneck block)

```cpp
constexpr int C = 8; // Channels for filters
constexpr int N = 2; // Batch size
constexpr int H = 3; // Height of input
constexpr int W = 3; // Width of input

int64_t xTensorDimA[]             = {N,  C, H,  W};
int64_t xTensorDimB[]             = {N,  C, H,  W};
int64_t xTensorDimC[]             = {N,  C, H,  W};
int64_t xTensorDimResidual[]      = {N,  C, H,  W};

int64_t wTensorDimA[]            = {C, C,  3,  3}; // Stack of 3x3 convolutions
int64_t wTensorDimB[]            = {C, C,  3,  3};
int64_t wTensorDimC[]            = {C, C,  3,  3};
int64_t wTensorResidual[]        = {C, C,  1,  1}; // Residual 1x1 convolution

int64_t yTensorDimA[]            = {N,  C, H,  W}; // Output sizes of convolutions
int64_t yTensorDimB[]            = {N,  C, H,  W};
int64_t yTensorDimC[]            = {N,  C, H,  W};
int64_t yTensorDimResidual[]     = {N,  C, H,  W};

int64_t perChannelScaleBiasDim[]      = { 1,  C, 1, 1};


int64_t sumTensorDim[]       = {1,  C, 1,  1};
int64_t sqSumTensorDim[]     = {1,  C, 1,  1};

int64_t conv_padA[]                = {1, 1};
int64_t conv_padB[]                = {1, 1};
int64_t conv_padC[]                = {1, 1};
int64_t conv_padResidual[]         = {0, 0};

int64_t conv_dilationA[]           = {1, 1};
int64_t conv_dilationB[]           = {1, 1};
int64_t conv_dilationC[]           = {1, 1};
int64_t conv_dilationResidual[]    = {1, 1};

int64_t conv_strideA[]             = {1, 1};
int64_t conv_strideB[]             = {1, 1};
int64_t conv_strideC[]             = {1, 1};
int64_t conv_strideResidual[]      = {1, 1};

// DP_FIRST_NODE (DP = direct path) is part of an enum ConvNode that allows for mapping between indicies to what convolution in the residual block

// you can set all of the conv filter sizes, paddings, strides, etc. if they choose. However, we can default to the original ResNet params if you doesn't pass those in. you are expected to pass in an input size though.
auto resnetBlockParamsBuilder = cudnn_frontend::ResNetBlockParamsBuilder()
                                            .setInputSize(xTensorDimA)
                                            .setConvFilterSizes({wTensorDimA, wTensorDimB, wTensorDimC, wTensorResidual})
                                            .setConvPaddings({conv_padA, conv_padB, conv_padC, conv_padResidual})
                                            .setConvStrides({conv_strideA, conv_strideB, conv_strideC, conv_strideResidual})
                                            .setConvDilations({conv_dilationA, conv_dilationB, conv_dilationC, conv_dilationResidual})
                                            .setConvDataTypeAt(cudnn_frontend::DP_FIRST_NODE, CUDNN_DATA_HALF); // Mixed precision


/** OR **/
auto resnetBlockParamsBuilder = cudnn_frontend::ResNetBlockParamsBuilder()
                                            .setInputSize(xTensorDimA)
                                            // Filter, pad, stride, etc. sizes default to ResNet paper params

// Set math precision
resnetBlockParamsBuilder.setMathPrec(CUDNN_DATA_FLOAT);

// Set the per channel scale and bias parameters
resnetBlockParamsBuilder.setScaleBiasDimsAt(cudnn_frontend::DP_FIRST_NODE, perChannelScaleBiasDim);

// Flags for the residual block. First flag is using legacy API in the backward block. Next flag is using FP8 precision in the block (defaults to FP16), and next flag is residual block specific, which is using 1x1 convolution in the residual path or not.
resnetBlockParamsBuilder.use_legacy_in_backward().use_fp8().use_1x1_conv();

// Build the params object
auto resnetBlockParams = resnetBlockParamsBuilder.build();

// Check if the builder was able to build the params object successfully
if (resnetBlockParams.getStatus() != CUDNN_STATUS_SUCCESS) {
    std::cout << resnetBlockParams.getErrorMessage() << std::endl;
    // Do something about error
}
```
As you can see, a user has the ability to set specific parameters for the block. The block also defaults to parameters of the original ResNet paper, allowing for ease of use for users that want to build ResNets identical to the original ResNet. This `params` object, in this case a `ResNetBlockParams` object, is passed into the construtor for the block. This object is SHARED between the forward and backward passes of the same block. Here's an example to construct a forward and backward residual bottleneck block. Because of the separate `params` object, the foward and backward block classes are encapsulated from one another, allowing for ease of isolation and debugging. However, the forward and backward passes need to communicate on what data to act on. This is where the `device pointer store` comes in.

## IMPORTANT: The params you need to set as the user
Internally, for parameters like convolution output and pooling output sizes, we calculate the expected output sizes internally and compare against yours. If you do not set convolution or pooling output sizes, we will set it for you. These params are common among the three blocks that you may want to set and their default values
- **Data type of the block and math precision of the block**. Defaults to mixed-precision FP16 and FP32 respectively (denoted as `CUDNN_DATA_HALF` and `CUDNN_DATA_FLOAT` respectively). For FP8 blocks, use `CUDNN_DATA_E4M3` or `CUDNN_DATA_E5M2.`

### Residual block
These are the params you may want to (and should for transparency as a user) set specifically for the residual block:

- **Each of the convolution input sizes, filter sizes, padding sizes, stride sizes, dilation sizes, and output sizes**. Note that if you just provide the initial input size and all the filters, paddings, stride, etc. for the convolutions, the block will calculate the rest of the input and output sizes for you (since the output of the a convolution in the block is the input to the next convolution in the block). However, for transparency, you probably want to set them yourself to ensure everything is the way you want it. Use `setInputSizes(), setFilterSizes()`, etc. to these parameters.
- **Per channel dimensions**. This is used for per channel scaling/biasing in batch normalization, as well as the `GEN STATS` node in the current implementation. Use `setBiasAndScaleDims()` to set. NOTE: If you set only the first node's per channel scale and bias dims, it will set the rest for you depending on the channels at that node. 
- **Flag to use a 1x1 convolution in the residual path.** Defaults to `true`. To not use, call `use_1x1_conv(false)` in the params builder.
- **Accumulation count for batch normalization.** This is calculated as  `N * H * W`  from the dims of the input to the batch normalization node. Use `setAccumCnt)` to set accumulation count. If not set, it does the calculation internally and sets it for each of the batch normalization nodes.

## Device Pointer Store
Each block also has a corresponding device pointer store to store GPU device pointers for the forward and backward pass to operate with. These device pointers are going to contain the actual numerical inputs and outputs for the block. you is expected to provide device pointers for inputs and outputs for each of the blocks and operations within the blocks (e.g. convolution, batch normalization). Setting up the device pointers is easy with our API, you just needs to provide the necessary device pointers and call the corresponding setters. This object is also SHARED between the forward and backward blocks of the same type. This is how the two passes communicate with each other. Here's an example with the same FP16 residual bottlneck block. Note that all necessary device pointers aren't seen for brevity:

```cpp
/** INITIALIZATION OF MOST DEVICE POINTERS OMITTED FOR BREVITY **/

// Example of how you might initialize a device pointer. Assume X0 is some arbitrary data structure that houses a size and device pointer
void* X0.devPtr = cudaMalloc((void**)&(X0.devPtr), X0.size() * sizeof(half)));

cudnn_frontend::ResNetBlockDevPtrStore devPtrStore; // Initialize a Residual block device pointer store

// Recall that residual bottlneck block has 4 convolutions (3 3x3 + 1 1x1 conv in residual path)
devPtrStore.setXDevPtrs({X0.devPtr, X1.devPtr, X2.devPtr, X3.devPtr}) // Set input convolution device pointers
            .setWDevPtrs({W0.devPtr, W1.devPtr, W2.devPtr, W3.devPtr}) // Set filter convolution device pointers
            .setYDevPtrs({Y0.devPtr, Y1.devPtr, Y2.devPtr, Y3.devPtr}) // Set output convolution device pointers
            .setFinalOutputDevPtr(finalOutput.devPtr); // Set final output device pointer

// Set batch normalization device pointers. Again, 4 BNs for each of the convolutions.
devPtrStore.setBNInScaleDevPtrs({scale.devPtr, scale.devPtr, scale.devPtr, scale.devPtr})
            .setBNInBiasDevPtrs({bias.devPtr, bias.devPtr, bias.devPtr, bias.devPtr})
            .setBNEqScaleDevPtrs({eq_scale0.devPtr, eq_scale1.devPtr, eq_scale2.devPtr, eq_scale3.devPtr})
            .setBNEqBiasDevPtrs({eq_bias0.devPtr, eq_bias1.devPtr, eq_bias2.devPtr, eq_bias3.devPtr})
            .setBNInMeanDevPtrs({in_mean.devPtr, in_mean.devPtr, in_mean.devPtr, in_mean.devPtr})
            .setBNInVarDevPtrs({in_var.devPtr, in_var.devPtr, in_var.devPtr, in_var.devPtr})
            .setBNOutMeanDevPtrs({out_mean.devPtr, out_mean.devPtr, out_mean.devPtr, out_mean.devPtr})
            .setBNOutVarDevPtrs({out_var.devPtr, out_var.devPtr, out_var.devPtr, out_var.devPtr})
            .setBNSavedMeanDevPtrs({saved_mean.devPtr, saved_mean.devPtr, saved_mean.devPtr, saved_mean.devPtr})
            .setBNSavedInvVarDevPtrs({saved_inv_var.devPtr, saved_inv_var.devPtr, saved_inv_var.devPtr, saved_inv_var.devPtr});

... 
...
...

cudaFree(X0.devPtr);
X0.devPtr = nullptr;
```
### **Device pointer memory allocation**
What's important is that the GPU device pointers you provide match, in total size, the datatype of the block. If you wanted mixed-precision FP16, you should be providing device pointers that have a total size of: (size of tensor) * sizeof(half). For example, suppose I have a 3x3 convolution filter with 1 input channel and 1 output channel. The dimensions might be [1, 1, 3, 3]. I'd have to allocate a device pointer of size 1 * 1 * 3 * 3 * sizeof(half) = 18 bytes. This is extremely important to ensure no page faults within the block, as internally, the block assumes that the configured block datatype matches the sizes of your provided device pointers.

## IMPORTANT: The device pointers you as the user need to set
### Residual block
These are the device pointers you need to set specifically for the residual block:

- **Convolution input, filter, and output device pointers for each of the convolutions**. Use `setXDevPtrs(), setWDevPtrs(), setYDevPtrs()` to these parameters.
- **Scale and bias device pointers**. This is used for per channel scaling/biasing in batch normalization. Use `setScaleDevPtr(), setBiasDevPtr()` to set.
- **Gen stats + Batch normalization device pointers**. Need `sum, sqSum, inScale, inBias, eqScale, eqBias, inMean, inVar, outMean, outInvVar, savedMean, savedInvVar` device pointers. There is a correspondeing setter for each of these. See sample for example.
- **Final output device pointer.** Final output device pointer after pooling. Use `setFinalOutputDevPtr()`.
- **Backward pass device pointers**. You need to set the following backward pass device pointers: `setdAfterReluGradDevPtr(), setdAfterMaxpoolGradDevPtr(), setdAfterBNGradDevPtr(), setdAfterConvDataGradDevPtr(), setdAfterConvWGrad(), setdBNScaleGradDevPtr(), setdBNBiasGradDevPtr(), setFinalOutputGradDevPtr()` 

## Building a block
Building a block is simple with our API, assuming you have configured the appropriate `params` structure. Building a block internally builds all the tensors, descriptors, operation graphs, and exeuction plans needed for the block. This is done for you. To build a block, depending on the block type (stem, residual, or classifier), you can call `createStemBlock()` or `createResidualBlock()` or `createClassifierBlock()` respectively with an initialzied `cudnnHandle_t`, a `string` of `"forward"` or `"backward"` (denoting if you want a forward block or backward block), a `std::shared_ptr` of type `IBlock`, and the corresponding `params`. Depending on the flags you set for the block, the `create...Block()` function will store into the pointer you passed in a built block with legacy API (if legacy flag was set in params), with FP8 (if FP8 flag was set in params), and in the case with the residual block, with 1x1 convolution in the residual path (if use_1x1_conv flag was set in params for residual block) and vice versa. Building the block returns a `cudnnStatus_t` denoting whether or not the build succeeded. If so, it returns `CUDNN_STATUS_SUCCESS`. If not, it will return the type of failure followed by a logged error message. You should check for the status for each block being built. Here's an example using our API to construct that same FP16 forward residual bottleneck block:

```cpp
cudnnHandle_t handle; // Create cudnn handle
checkCudnnErr(cudnnCreate(&handle));

auto resnetBlockParamsBuilder = ... /** configure parameters **/
// Build the params object
auto resnetBlockParams = resnetBlockParamsBuilder.build();

// Creates Residual Block with params. We use std::shared_ptr to initialize the block pointer
std::shared_ptr<IBlock> residualForwardBlock;

// We want a residual forward block, so we call createResidualBlock() with the "forward" string and the params object and the IBlock ptr
// Builds operation graphs and execution plans for the block. 
cudnnStatus_t buildStatus = createResidualBlock(handle, "forward", residualForwardBlock, params);

if (buildStatus != CUDNN_STATUS_SUCCESS) {
    std::cout << residualBlock.getErrorMessage() << std::endl;
    /** Do something about error **/    
}
```

## Exeucting a block
Once you have a built block, you can execute the forward or backward pass (depending on the block type) now! In cuDNN, in order to execute an operation graph, a `variant pack` needs to be supplied to the graph. We handle all the `variant pack` creation internally. Once all the variant packs are succesfully created, the graph can be executed. We handle this all internally, and you as the user will just need to call our simple API `runBlock()`. All you need to do is pass the `device pointer store` you configured earlier to our `runBlock()` API for each of the blocks. `runBlock()` creates the variant packs and executes the block pass. This will return a `cudnnStatus_t` to indicate whether or not the variant packs were created successfully and whether or not the forward/backward pass was executed successfully (you should check for this). Here's an example with the same FP16 forward residual bottleneck block.
```cpp
cudnnHandle_t handle; // Create cudnn handle
checkCudnnErr(cudnnCreate(&handle));

auto resnetBlockParamsBuilder = ... /** configure parameters **/
// Build the params object
auto resnetBlockParams = resnetBlockParamsBuilder.build();

// Creates Residual Block with params. We use std::shared_ptr to initialize the block pointer
std::shared_ptr<IBlock> residualForwardBlock;

// We want a residual forward block, so we call createResidualBlock() with the "forward" string and the params object and the IBlock ptr
// Builds operation graphs and execution plans for the block. 
cudnnStatus_t buildStatus = createResidualBlock(handle, "forward", residualForwardBlock, params);

if (buildStatus != CUDNN_STATUS_SUCCESS) {
    std::cout << residualBlock.getErrorMessage() << std::endl;
    /** Do something about error **/    
}

cudnn_frontend::ResNetBlockDevPtrStore devPtrStore; // Initialize a Residual block device pointer store

/** Configure all the device pointers **/
...
...
...

// Creates variant packs based on devPtrStore and executes
status = runBlock(handle, residualForwardBlock, devPtrStore);

if (status != CUDNN_STATUS_SUCCESS) {
    std::cout << residualForwardBlock->getErrorMessage() << std::endl;
    /** Do something about error **/
}
``` 

## Running an entire network
To create a ResNet network like a ResNet50, it's quite easy with these blocks. First, within a ResNet, there are the two blocks the stem and classifier block, and in the middle there are stacks of residual/bottleneck blocks. So in total for a ResNet50, the user will only need to actually create the Stem Block, the Residual Block, and the classifier block once. The user may need to create 1-3 different residual blocks depending on the number of filters in the block, but the whole point is that **you only have to create a block once** and you can execute different data on the block. Suppose for a dummy example we have a ResNet50 with a stem block that takes in 3x224x224 input, and outputs probabilities for 100 classes. Each intermediate block will have 64 channels with a constant 56x56 dimension like the original ResNet paper. The only thing the user needs to do is configure a device pointer store 50 different times for the input and output data for each of the blocks as opposed to configuring 50 different parameters, creating 50 blocks, configuring 50 different device pointer stores, and executing 50 different times. In summary, in this example, the user would need to create a single stem block, a single residual block, and a single classifier block. The user will also need to configure a stem block device pointer store, 50 residual block device pointer stores, and a classifier block device pointer store. To create and run this network using our blocks, it may look something like this:

```cpp
int NUM_BLOCKS = 25;
cudnnHandle_t handle;
checkCudnnErr(cudnnCreate(&handle));


int NUM_BLOCKS = 6;
// Creates Residual Block with params
std::shared_ptr<IBlock> residualBlock;

// Create a residual block. Only need to create once if identical blocks.
status = createResidualBlock(handle, "forward", residualBlock, residualBlockParams);

if (status != CUDNN_STATUS_SUCCESS) {
    if (residualBlock == nullptr && status == CUDNN_STATUS_NOT_SUPPORTED) {
        std::cout << "[ERROR]: Block type not supported" << std::endl;
        /** Do something about error **/
        return;
    }
    std::cout << residualBlock->getErrorMessage() << std::endl;
    /** Do something about error **/
    return;
}

for (int block = 0; layer < NUM_BLOCKS; block++) {
    std::cout << "EXECUTING BLOCK " << block << std::endl;
    std::cout << "=============================" << std::endl;

    // Creates variant packs based on devPtrStore and executes. Uses the same created block but can execute with differnt data. Assuming the user has created NUM_BLOCKS different dev ptr stores in a vector
    status = runBlock(handle, residualBlock, residualBlockDevPtrStores[block]);

    if (status != CUDNN_STATUS_SUCCESS) {
        std::cout << residualBlock->getErrorMessage() << std::endl;
        /** Do something about error **/
    }
}

```

## Debugging
Internally, the block checks the statuses for all tensor, descriptor, operation graph, and execution plan creations. If one of these objects fails to build, the block will notify you of the failure and store the error message. You should run with `CUDNN_FRONTEND_LOG_INFO=1` and `CUDNN_FRONTEND_LOG_FILE=file/to/log/to.txt` for logged info, warning, and error messages. 

### Running with NV_CUDNN_DISABLE_EXCEPTION
If you're running with the flag `#NV_CUDNN_DISABLE_EXCEPTION`, then no exceptions will be thrown in the block if an error happens. Instead, if an error is caught within the block, the block will return a status that is not `CUDNN_STATUS_SUCCESS` and store the error message internally. You can use `block->getErrorMessage()` like above to get the message.