/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "../utils/helpers.h"
#include <cmath>

template <typename T_ELEM>
void
weightGrad_cpu_ref(const T_ELEM* image,
                   const T_ELEM* diffData,
                   T_ELEM* output,
                   cudnnTensorFormat_t filterFormat,
                   const int64_t* inDims,
                   const int64_t* filDims,
                   const int64_t* diffDims,
                   const int64_t* stride,
                   const int64_t* pad,
                   const int64_t* dilation,
                   int nbDims) {
    float alpha = 1.0f;
    float beta  = 0.0;
    // Some sanity checks
    // image   is n x c x h x w
    // diff    is n x k x p x q
    // filter  is k x c x r x s
    assert(inDims[0] == diffDims[0]);
    assert(inDims[1] == filDims[1]);
    assert(diffDims[1] == filDims[0]);

    // Filter stride
    int64_t filterStride[8];
    int64_t inStride[8];
    int64_t diffStride[8];

    generateStrides(inDims, inStride, nbDims, filterFormat);
    generateStrides(diffDims, diffStride, nbDims, filterFormat);
    generateStrides(filDims, filterStride, nbDims, filterFormat);

    bool isConv = true;  //(CUDNN_CONVOLUTION == mode) ;

    // For every filter pixel (k x c x r x s)
    for (size_t ci = 0; ci < (size_t)inDims[1]; ci++) {               // Loop over filter output pixels
        for (size_t ri = 0; ri < (size_t)filDims[2]; ri++) {          //        ^
            for (size_t si = 0; si < (size_t)filDims[3]; si++) {      //    ^
                for (size_t ki = 0; ki < (size_t)filDims[0]; ki++) {  // ^
                    int64_t filIdx =
                        ki * filterStride[0] + ci * filterStride[1] + ri * filterStride[2] + si * filterStride[3];
                    float val = 0.f;
                    // For every image (n)
                    for (int ni = 0; ni < inDims[0]; ni++) {  // Sum over the batch
                        int64_t offset_image = ni * inStride[0] + ci * inStride[1];
                        int64_t offset_diff  = ni * diffStride[0] + ki * diffStride[1];
                        // For every pixel in diff (p x q)
                        for (int pi = 0; pi < diffDims[2]; pi++) {      // Sum over the pixels of diff
                            for (int qi = 0; qi < diffDims[3]; qi++) {  //  ^
                                // Fetch the value in image and diff, product and accumulate
                                int64_t y = pi * stride[0] - pad[0];
                                int64_t x = qi * stride[1] - pad[1];
                                // Convolution = Correlation with a flipped filter
                                // So basically, for the convolution, we replace r by dim-1-r
                                // and s by dim-1-s to "flip" the filter.
                                // We can then just reason in term of correlation
                                if (isConv) {
                                    y += (filDims[2] - 1 - ri) * dilation[0];
                                    x += (filDims[3] - 1 - si) * dilation[1];
                                } else {
                                    // The effect of dilation on the gradient is to start
                                    // the "zone of influence" of a given pixel further
                                    // into the image, so dilation
                                    // only produces a shift in x and y
                                    y += ri * dilation[0];
                                    x += si * dilation[1];
                                }
                                // Image value
                                int64_t inBounds = ((x >= 0) && (x < inDims[3]) && (y >= 0) && (y < inDims[2]));
                                if (inBounds) {
                                    int imIdx = static_cast<int>(offset_image + y * inStride[2] + x * inStride[3]);
                                    // Diff value
                                    int diffIdx =
                                        static_cast<int>(offset_diff + pi * diffStride[2] + qi * diffStride[3]);
                                    // Prod and accumulate
                                    T_ELEM imTmp   = image[imIdx];
                                    T_ELEM diffTmp = diffData[diffIdx];
                                    val            = doFma(diffTmp, imTmp, val);
                                }
                            }
                        }
                    }
                    doEpilog(output, filIdx, alpha * val, beta);
                }
            }
        }
    }
}

template <typename scale_bias_type = float>
void
scale_and_bias_tensor_cpu(const half* inputData,
                          float* outputData,
                          const scale_bias_type* scaleData,
                          const scale_bias_type* biasData,
                          const int64_t inputSize,
                          const int64_t* inputDims) {
    // Scale and bias per channel basis. Assumes NHWC format.
    for (int i = 0; i < inputSize; i++) {
        int c = i % inputDims[0];
        if constexpr (std::is_same_v<scale_bias_type, float>) {
            outputData[i] = (__half2float(inputData[i]) * scaleData[c] + biasData[c]);
        } else {
            outputData[i] = (__half2float(inputData[i]) * __half2float(scaleData[c]) + __half2float(biasData[c]));
        }
    }
}

template <typename inputType>
void
add_tensors_cpu(const inputType* firstInputData,
                const inputType* secondInputData,
                inputType* outputData,
                const int64_t inputSize) {
    for (int i = 0; i < inputSize; i++) {
        outputData[i] = firstInputData[i] + secondInputData[i];
    }
}

template <typename outputType>
void
relu(const float* inputData, outputType* outputData, const int64_t inputSize) {
    for (int i = 0; i < inputSize; i++) {
        float output = inputData[i] > 0.0f ? inputData[i] : 0.0f;
        if constexpr (std::is_same_v<outputType, float>) {
            outputData[i] = output;
        } else {
            outputData[i] = __float2half(output);
        }
    }
}

void
gen_stats_cpu(const half* inputData,
              std::vector<std::pair<float, float>>& outputData,
              const int64_t inputSize,
              const int64_t* inputDims) {
    int64_t channel_dim = inputDims[1];
    std::vector<int64_t> totals((size_t)channel_dim, 0);
    for (int i = 0; i < inputSize; i++) {
        int channel_index = i % channel_dim;

        // Sum
        outputData[channel_index].first = outputData[channel_index].first + __half2float(inputData[i]);
        totals[channel_index]           = totals[channel_index] + 1;
    }

    // Calculate the mean for each channel. Assumes NHWC format.
    for (size_t i = 0; i < outputData.size(); i++) {
        outputData[i].first = outputData[i].first / totals[i];
    }

    for (int i = 0; i < inputSize; i++) {
        int channel_index = i % channel_dim;

        // Sum of squares
        float diff = (__half2float(inputData[i]) - outputData[channel_index].first) *
                     (__half2float(inputData[i]) - outputData[channel_index].first);
        outputData[channel_index].second = outputData[channel_index].second + diff;
    }

    // Calculate the variance for the channel. Assumes NHWC format.
    for (size_t i = 0; i < outputData.size(); i++) {
        outputData[i].second = outputData[i].second / totals[i];
    }
}

void
batch_normalize(const half* inputData,
                half* outputData,
                const std::vector<std::pair<float, float>>& stats,
                const int64_t inputSize,
                const int64_t* inputDims) {
    int64_t channel_dim = inputDims[1];
    // Loop through each element in the input and normalize it based on what batch it belongs to
    for (int i = 0; i < inputSize; i++) {
        int batch_index = i % channel_dim;
        outputData[i]   = __float2half((__half2float(inputData[i]) - stats[batch_index].first) /
                                     (float)std::sqrt(stats[batch_index].second));
    }
}

// T_ELEM is the type the data is stored in, T_MATH is the type the calculations are done in.
template <typename T_ELEM, typename T_MATH>
void
conv_cpu_ref(const T_ELEM* inputData,
             const T_ELEM* filterData,
             T_ELEM* outputData,
             int resizeFactor,
             cudnnTensorFormat_t filterFormat,
             const int64_t* inDims,
             const int64_t* filDims,
             const int64_t* diffDims,
             const int64_t* stride,
             const int64_t* pad,
             const int64_t* dilation,
             int64_t nbDims) {
    int64_t imDims = nbDims - 2;
    float alpha    = 1.0f;
    float beta     = 0.0;
    // Some sanity checks
    // image   is n x c x h x w
    // diff    is n x k x p x q
    // filter  is k x c x r x s
    assert(inDims[0] == diffDims[0]);
    assert(inDims[1] == filDims[1]);
    assert(diffDims[1] == filDims[0]);

    // Filter stride
    int64_t filterStride[8];
    int64_t inStride[8];
    int64_t diffStride[8];

    generateStrides(inDims, inStride, nbDims, filterFormat);
    generateStrides(diffDims, diffStride, nbDims, filterFormat);
    generateStrides(filDims, filterStride, nbDims, filterFormat);

    int64_t filStride[8] = {0};
    generateStrides(filDims, filStride, nbDims, filterFormat);

    bool isConv = true;  //(CUDNN_CONVOLUTION == mode) ;

    // Number of pixels in output
    int64_t nPixelsOut = 1;
    for (int i = 2; i < nbDims; i++) {
        nPixelsOut *= diffDims[i];
    }

    // Number of pixels in filter
    int64_t nPixelsFil = 1;
    for (int i = 2; i < nbDims; i++) {
        nPixelsFil *= filDims[i];
    }

    // Used to store coordinates
    int64_t filIds[8] = {0};
    int64_t outIds[8] = {0};
    int64_t inIds[8]  = {0};
    int64_t tmpIds[8] = {0};

    // For each image in the output
    for (int64_t ni = 0; ni < diffDims[0]; ni++) {
        // For each outer feature layer of the output image
        for (int ki_outer = 0; ki_outer < diffDims[1] / resizeFactor; ki_outer++) {
            int64_t outputOffset = ni * diffStride[0] / resizeFactor + ki_outer * diffStride[1];
            // For every pixel in this output image's feature layer
            for (int outId = 0; outId < nPixelsOut; outId++) {
                // Get output pixel ids
                lin2dim(outId, outIds, diffDims + 2, imDims);  // Skip n and k dimensions
                // Now we get the coordinates in input space of the "top left" corner
                // of the filter: multiply by stride and remove pad
                for (int d = 0; d < imDims; d++) {
                    inIds[d] = outIds[d] * stride[d] - pad[d];
                }
                // For each inner feature layer of the output image
                for (int ki_inner = 0; ki_inner < resizeFactor; ki_inner++) {
                    // We prepare to accumulate
                    T_MATH tmp = 0;
                    // For each outer feature layer of the input image and filter
                    for (int ci = 0; ci < inDims[1] / resizeFactor; ci++) {
                        int64_t inputOffset = ni * inStride[0] / resizeFactor + ci * inStride[1];
                        int64_t filterOffset =
                            (ki_outer * resizeFactor + ki_inner) * filStride[0] / resizeFactor + ci * filStride[1];
                        // Now for every pixel in the filter
                        for (int filId = 0; filId < nPixelsFil; filId++) {
                            // Get the position of the pixel
                            lin2dim(filId, filIds, filDims + 2, imDims);
                            // Compute the corresponding output pixel
                            // and check whether we are in the padding area on the fly too
                            // (not that for convolution, we flip the image patch;
                            // equivalent to flipping the filter patch).
                            bool inside = true;
                            for (int d = 0; d < imDims && inside; d++) {
                                if (isConv) {
                                    tmpIds[d] = inIds[d] + dilation[d] * (filDims[2 + d] - 1 - filIds[d]);
                                } else {
                                    tmpIds[d] = inIds[d] + dilation[d] * filIds[d];
                                }
                                // If we are in the padding area: stop and skip computations
                                inside &= (tmpIds[d] >= 0 && tmpIds[d] < inDims[2 + d]);
                            }
                            if (inside) {
                                int64_t actualTmpId = inputOffset + dim2lin(tmpIds, (inStride) + 2, imDims);
                                // int actualFilId = filterOffset + filId ;
                                int64_t actualFilId = filterOffset + dim2lin(filIds, (filStride) + 2, imDims);

                                // For each inner feature layer of the input image and filter
                                for (int i = 0; i < resizeFactor; i++) {
                                    T_ELEM fval = filterData[(size_t)(actualFilId * resizeFactor + i)];
                                    T_ELEM ival = inputData[(size_t)(actualTmpId * resizeFactor + i)];
                                    tmp         = doFma(fval, ival, tmp);
                                }
                            }
                        }
                    }

                    // Store final result in proper position in output image
                    int64_t actualOutId = outputOffset + dim2lin(outIds, (diffStride) + 2, imDims);
                    doEpilog(outputData, actualOutId * resizeFactor + ki_inner, alpha * tmp, beta);
                }
            }
        }
    }
}

template <typename T_ELEM>
void
dataGrad_cpu_ref(const T_ELEM* weight,
                 const T_ELEM* top_diff,
                 T_ELEM* output,
                 cudnnTensorFormat_t filterFormat,
                 const int64_t* inDims,
                 const int64_t* filDims,
                 const int64_t* outDims,
                 const int64_t* stride,
                 const int64_t* pad,
                 const int64_t* dilation,
                 int nbDims,
                 cudnnConvolutionMode_t mode) {
    // Sanity checks
    // output is n x c x h x w
    // diff   is n x k x p x q
    // filter is k x c x r x s
    assert(inDims[0] == outDims[0]);   // n
    assert(inDims[1] == filDims[0]);   // k
    assert(outDims[1] == filDims[1]);  // cactualOutId

    int64_t inStride[8];
    int64_t outStride[8];

    float alpha = 1.0f;
    float beta  = 0.0;

    generateStrides(inDims, inStride, nbDims, filterFormat);
    generateStrides(outDims, outStride, nbDims, filterFormat);

    int64_t filStride[8] = {0};
    generateStrides(filDims, filStride, nbDims, filterFormat);

    // true for convolution and false for cross-correlation
    bool isConv = (mode == CUDNN_CONVOLUTION) ? true : false;

    // For every output pixel (n x c x h x w)
    for (int ni = 0; ni < outDims[0]; ni++) {
        for (int ci = 0; ci < outDims[1]; ci++) {
            for (int hi = 0; hi < outDims[2]; hi++) {
                for (int wi = 0; wi < outDims[3]; wi++) {
                    int64_t outIdx = ni * outStride[0] + ci * outStride[1] + hi * outStride[2] + wi * outStride[3];
                    float val      = 0.0;

                    // For every diff channel (k)
                    for (int ki = 0; ki < inDims[1]; ki++) {  // Sum over k channels
                        int64_t offset_filter = ki * filStride[0] + ci * filStride[1];
                        int64_t offset_diff   = ni * inStride[0] + ki * inStride[1];
                        // For every pixel if filter (r x s)
                        for (int ri = 0; ri < filDims[2]; ri++) {
                            int64_t p = hi + pad[0];

                            if (isConv) {
                                p -= (filDims[2] - 1 - ri) * dilation[0];
                            } else {
                                p -= ri * dilation[0];
                            }

                            if (p % stride[0]) {
                                continue;
                            }

                            p /= stride[0];

                            for (int si = 0; si < filDims[3]; si++) {
                                int64_t q = wi + pad[1];

                                // Fetch the value in filter and diff, product and accumulate
                                // So basically, for the convolution, we replace r by dim-1-r
                                // and s by dim-1-s to "flip" the filter
                                // We can then just reason in term of correlation
                                if (isConv) {
                                    q -= (filDims[3] - 1 - si) * dilation[1];
                                } else {
                                    q -= si * dilation[1];
                                }

                                // Skip if q or p isn't multiple of strides
                                if (q % stride[1]) {
                                    continue;
                                }

                                q /= stride[1];

                                int inBounds = ((p >= 0) && (p < inDims[2]) && (q >= 0) && (q < inDims[3]));
                                if (inBounds) {
                                    int64_t filterIdx = offset_filter + ri * filStride[2] + si * filStride[3];
                                    int64_t diffIdx   = offset_diff + p * inStride[2] + q * inStride[3];
                                    T_ELEM imTmp      = top_diff[(size_t)diffIdx];
                                    T_ELEM filTmp     = weight[(size_t)filterIdx];
                                    val               = doFma(filTmp, imTmp, val);
                                }
                            }
                        }
                    }
                    doEpilog(output, outIdx, alpha * val, beta);
                }
            }
        }
    }
}
