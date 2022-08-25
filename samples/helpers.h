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
#include <iostream>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "catch.hpp"

#include <cudnn.h>

#include "fp16_dev.h"
#include "fp16_emu.h"

#define THRESHOLD 2.0e-2

int getFwdConvDilatedFilterDim(int filterDim, int dilation);
int getFwdConvPaddedImageDim(int tensorDim, int pad);
int getFwdConvOutputDim( int tensorDim, int pad, int filterDim, int stride, int dilation);

void generateStrides(const int64_t* dimA, int64_t* strideA, int nbDims, cudnnTensorFormat_t filterFormat);

int checkCudaError(cudaError_t code, const char* expr, const char* file, int line);
int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line);

void lin2dim(int id, int64_t* ids, const int64_t* dims, int length);
int dim2lin(const int64_t* ids, const int64_t* strides, int length);


void initImage(float* image, int64_t imageSize);
void initImage(half1* image, int64_t imageSize);
void initImage(int8_t* image, int imageSize);
void initImage(int32_t* image, int imageSize);
void initImage(int64_t* image, int imageSize);
void initImage(bool* image, int imageSize);
void initImagePadded(int8_t* image, int64_t dimA[], int64_t dimPadded[], int64_t stridePadded[], cudnnDataType_t dataType);

void doEpilog(float* out, int idx, float alphaAcc, float beta);
void doEpilog(half1* out, int idx, float alphaAcc, float beta);
void doEpilog(int8_t* out, int idx, int32_t alphaAcc, float beta);


float getError(float dev, float ref);
float getError(half1 dev, half1 ref);
int8_t getError(int8_t dev, int8_t ref);

static float doFma(float fval, float ival, float tmp) {
    return fval * ival + tmp;
}

static float doFma(half1 fval, half1 ival, float tmp) {
    return cpu_half2float(fval) * cpu_half2float(ival) + tmp;
}

static int32_t doFma(int8_t fval, int8_t ival, int32_t tmp) {
    return int32_t(fval) * int32_t(ival) + tmp;
}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static int32_t doFma(float fval, float ival, int32_t tmp) {
    (void)fval;
    (void)ival;
    (void)tmp;
    return 0;
}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static int32_t doFma(half1 fval, half1 ival, int32_t tmp) {
    (void)fval;
    (void)ival;
    (void)tmp;
    return 0;
}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static float doFma(int8_t fval, int8_t ival, float tmp) {
    (void)fval;
    (void)ival;
    (void)tmp;
    return 0;
}

#define checkCudaErr(...)                                                        \
    do {                                                                         \
        int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        REQUIRE(err == 0);                                                       \
    } while (0)

#define checkCudnnErr(...)                                                        \
    do {                                                                          \
        int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        REQUIRE(err == 0);                                                        \
    } while (0)



template <typename T_ELEM>
class SurfaceManager {

public:
    T_ELEM* devPtrX = NULL;
    T_ELEM* devPtrW = NULL;
    T_ELEM* devPtrY = NULL;
    T_ELEM* devPtrZ = NULL;
    T_ELEM* devPtrB = NULL;
    T_ELEM* devPtrAfterAdd = NULL;
    T_ELEM* devPtrAfterConv = NULL;
    T_ELEM* devPtrAfterBias = NULL;

    T_ELEM* hostX          = NULL;
    T_ELEM* hostW          = NULL;
    T_ELEM* hostY          = NULL;
    T_ELEM* hostZ          = NULL;
    T_ELEM* hostB          = NULL;
    T_ELEM* hostAfterAdd   = NULL;
    T_ELEM* hostAfterConv  = NULL;
    T_ELEM* hostAfterBias  = NULL;
    T_ELEM* host_ref = NULL;


explicit SurfaceManager(int64_t Xsize, int64_t Wsize, int64_t Ysize, int ref_size) {
    checkCudaErr(cudaMalloc((void**)&(devPtrX), (Xsize) * sizeof(devPtrX[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrW), (Wsize) * sizeof(devPtrW[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrY), (Ysize) * sizeof(devPtrY[0])));

    hostX     = (T_ELEM*) calloc(Xsize, sizeof(hostX[0]));
    hostW     = (T_ELEM*) calloc(Wsize, sizeof(hostW[0]));
    hostY     = (T_ELEM*) calloc(Ysize, sizeof(hostY[0]));
    host_ref  = (T_ELEM*) calloc(ref_size, sizeof(host_ref[0]));

    initImage(hostX, Xsize);
    initImage(hostW, Wsize);
    initImage(hostY, Ysize);

    checkCudaErr(cudaMemcpy(devPtrX, hostX, sizeof(hostX[0]) * Xsize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrW, hostW, sizeof(hostW[0]) * Wsize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrY, hostY, sizeof(hostY[0]) * Ysize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());
}

explicit SurfaceManager(int64_t Xsize, int64_t Wsize, int64_t Ysize, int64_t Bsize, bool isConvBiasAdd) {
    (void)isConvBiasAdd;

    checkCudaErr(cudaMalloc((void**)&(devPtrX), (Xsize) * sizeof(devPtrX[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrW), (Wsize) * sizeof(devPtrW[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrY), (Ysize) * sizeof(devPtrY[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrZ), (Ysize) * sizeof(devPtrZ[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrB), (Bsize) * sizeof(devPtrB[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrAfterConv), (Ysize) * sizeof(devPtrAfterConv[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrAfterAdd), (Ysize) * sizeof(devPtrAfterAdd[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrAfterBias), (Ysize) * sizeof(devPtrAfterBias[0])));

    hostX     = (T_ELEM*) calloc(Xsize, sizeof(hostX[0]));
    hostW     = (T_ELEM*) calloc(Wsize, sizeof(hostW[0]));
    hostY     = (T_ELEM*) calloc(Ysize, sizeof(hostY[0]));
    hostZ     = (T_ELEM*) calloc(Ysize, sizeof(hostZ[0]));
    hostB     = (T_ELEM*) calloc(Bsize, sizeof(hostB[0]));
    hostAfterConv = (T_ELEM*) calloc(Ysize, sizeof(hostAfterConv[0]));
    hostAfterAdd  = (T_ELEM*) calloc(Ysize, sizeof(hostAfterAdd[0]));
    hostAfterBias = (T_ELEM*) calloc(Ysize, sizeof(hostAfterBias[0]));
    host_ref  = (T_ELEM*) calloc(Ysize, sizeof(host_ref[0]));

    initImage(hostX, Xsize);
    initImage(hostW, Wsize);
    initImage(hostY, Ysize);
    initImage(hostZ, Ysize);
    initImage(hostB, Bsize);
    initImage(hostAfterAdd, Ysize);
    initImage(hostAfterBias, Ysize);
    initImage(hostAfterConv, Ysize);
    
    checkCudaErr(cudaMemcpy(devPtrX, hostX, sizeof(hostX[0]) * Xsize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrW, hostW, sizeof(hostW[0]) * Wsize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrY, hostY, sizeof(hostY[0]) * Ysize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrZ, hostZ, sizeof(hostZ[0]) * Ysize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrB, hostB, sizeof(hostB[0]) * Bsize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrAfterAdd, hostAfterAdd, sizeof(hostAfterAdd[0]) * Ysize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrAfterBias, hostAfterBias, sizeof(hostAfterBias[0]) * Ysize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrAfterConv, hostAfterConv, sizeof(hostAfterConv[0]) * Ysize, cudaMemcpyHostToDevice));

    checkCudaErr(cudaDeviceSynchronize());
}

~SurfaceManager() {
    if (devPtrX) cudaFree(devPtrX);
    if (devPtrW) cudaFree(devPtrW);
    if (devPtrY) cudaFree(devPtrY);
    if (devPtrZ) cudaFree(devPtrZ);
    if (devPtrB) cudaFree(devPtrB);
    if (devPtrAfterAdd) cudaFree(devPtrAfterAdd);
    if (devPtrAfterBias) cudaFree(devPtrAfterBias);
    if (devPtrAfterConv) cudaFree(devPtrAfterConv);

    if (hostX) free(hostX);
    if (hostW) free(hostW);
    if (hostY) free(hostY);
    if (hostZ) free(hostZ);
    if (hostB) free(hostB);
    if (hostAfterAdd) free(hostAfterAdd);
    if (hostAfterBias) free(hostAfterBias);
    if (hostAfterConv) free(hostAfterConv);
    if (host_ref) free(host_ref);
}

};



template <typename T_ELEM>
struct Surface {
    T_ELEM* devPtr = NULL;
    T_ELEM* hostPtr = NULL;
    T_ELEM* hostRefPtr = NULL;

    explicit Surface(int64_t n_elems, bool hasRef) {
        checkCudaErr(cudaMalloc((void**)&(devPtr), (n_elems) * sizeof(devPtr[0])));
        hostPtr = (T_ELEM*) calloc(n_elems, sizeof(hostPtr[0]));
        if(hasRef) {
            hostRefPtr = (T_ELEM*) calloc(n_elems, sizeof(hostRefPtr[0]));
        }
        initImage(hostPtr, n_elems);
        checkCudaErr(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());
    }

    explicit Surface(int64_t n_elems, bool hasRef, bool isInterleaved) {
        checkCudaErr(cudaMalloc((void**)&(devPtr), (n_elems) * sizeof(devPtr[0])));
        hostPtr = (T_ELEM*) calloc(n_elems, sizeof(hostPtr[0]));
        if(hasRef) {
            hostRefPtr = (T_ELEM*) calloc(n_elems, sizeof(hostRefPtr[0]));
        }
        initImage(hostPtr, n_elems);
	uint32_t *temp = (uint32_t *)hostPtr;
	for (int i = 0; i < n_elems; i = i+2) {
	    temp[i + 1] = 1u;
	}
        checkCudaErr(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());
    }

    ~Surface() {
        if (devPtr) cudaFree(devPtr);
        if (hostPtr) free(hostPtr);
        if (hostRefPtr) free(hostRefPtr);
    }

};
