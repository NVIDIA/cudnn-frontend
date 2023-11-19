/*
 * Copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "cudnn_frontend.h"

#if (CUDNN_VERSION >= 8800)
using namespace cudnn_frontend;
#include "resnet_sample.h"

TEST_CASE("Residual block without residual conv", "[Resnet][Residual]") {
    int64_t const N = 400;
    int64_t const C = 256;
    int64_t const H = 56;
    int64_t const W = 56;

    int64_t const K[]      = {64, 64, 256};
    int64_t const R[]      = {1, 3, 1};
    int64_t const S[]      = {1, 3, 1};
    int64_t const pad[]    = {0, 1, 0};
    int64_t const stride[] = {1, 1, 1};

    /* Start of device Pointer Creation */

    // Forward
    using fp8_e4m3 = int8_t;

    Surface<fp8_e4m3> X(N * C * H * W, false);

    Surface<float> XDescale0(1, false);
    Surface<float> XDescale1(1, false);
    Surface<float> XDescale2(1, false);

    Surface<fp8_e4m3> WeightNHWC0(K[0] * C * R[0] * S[0], false);
    Surface<fp8_e4m3> WeightNHWC1(K[1] * K[0] * R[1] * S[1], false);
    Surface<fp8_e4m3> WeightNHWC2(K[2] * K[1] * R[2] * S[2], false);

    Surface<float> WDescale0(1, false);
    Surface<float> WDescale1(1, false);
    Surface<float> WDescale2(1, false);

    Surface<fp8_e4m3> Y0(N * K[0] * H * W, false);
    Surface<fp8_e4m3> Y1(N * K[1] * H * W, false);
    Surface<fp8_e4m3> Y2(N * K[2] * H * W, false);

    Surface<float> YScale0(1, false);
    Surface<float> YScale1(1, false);
    Surface<float> YScale2(1, false);

    Surface<float> ZDescale(1, false);

    Surface<float> YAmax0(1, false);
    Surface<float> YAmax1(1, false);
    Surface<float> YAmax2(1, false);

    Surface<float> BNXDescale0(1, false);
    Surface<float> BNXDescale1(1, false);
    Surface<float> BNXDescale2(1, false);

    Surface<fp8_e4m3> BNY0(N * K[0] * H * W, false);
    Surface<fp8_e4m3> BNY1(N * K[1] * H * W, false);
    Surface<fp8_e4m3> BNY2(N * K[2] * H * W, false);

    Surface<float> BNYAmax0(1, false);
    Surface<float> BNYAmax1(1, false);
    Surface<float> BNYAmax2(1, false);

    Surface<float> BNYScale0(1, false);
    Surface<float> BNYScale1(1, false);
    Surface<float> BNYScale2(1, false);

    Surface<float> BN_running_mean0(K[0], false);
    Surface<float> BN_running_mean1(K[1], false);
    Surface<float> BN_running_mean2(K[2], false);

    Surface<float> BN_running_var0(K[0], false);
    Surface<float> BN_running_var1(K[1], false);
    Surface<float> BN_running_var2(K[2], false);

    Surface<float> BN_saved_mean0(K[0], false);
    Surface<float> BN_saved_mean1(K[1], false);
    Surface<float> BN_saved_mean2(K[2], false);

    Surface<float> BN_saved_inv_var0(K[0], false);
    Surface<float> BN_saved_inv_var1(K[1], false);
    Surface<float> BN_saved_inv_var2(K[2], false);

    Surface<float> BN_scale0(K[0], false);
    Surface<float> BN_scale1(K[1], false);
    Surface<float> BN_scale2(K[2], false);

    Surface<float> BN_bias0(K[0], false);
    Surface<float> BN_bias1(K[1], false);
    Surface<float> BN_bias2(K[2], false);

    std::vector<float> epsilons        = {0.05f, 0.05f, 0.05f};
    std::vector<float> exp_avg_factors = {0.9f, 0.9f, 0.9f};

    /* End of device Pointer Creation */

    // Set the convolution parameters
    auto residualBlockParams = cudnn_frontend::ResidualBlockParamsBuilder()
                                   .setInputDim({N, C, H, W})
                                   .setConvolutionParams(cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO,
                                                         {K[0], R[0], S[0], pad[0], stride[0]})
                                   .setConvolutionParams(cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE,
                                                         {K[1], R[1], S[1], pad[1], stride[1]})
                                   .setConvolutionParams(cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO,
                                                         {K[2], R[2], S[2], pad[2], stride[2]})
                                   .build();

    ResidualBlockDevPtrStore devPtrStore;

    // Set device ptrs
    devPtrStore.setXDevPtr(X.devPtr)
        .setYDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, Y0.devPtr},
                      {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, Y1.devPtr},
                      {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, Y2.devPtr}})
        .setWeightNHWCDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, WeightNHWC0.devPtr},
                               {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, WeightNHWC1.devPtr},
                               {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, WeightNHWC2.devPtr}})
        .setBNXDescaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BNXDescale0.devPtr},
                               {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BNXDescale1.devPtr},
                               {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BNXDescale2.devPtr}})
        .setBNYDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BNY0.devPtr},
                        {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BNY1.devPtr},
                        {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BNY2.devPtr}})
        .setBNYScaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BNYScale0.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BNYScale1.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BNYScale2.devPtr}})
        .setBNYAMaxDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BNYAmax0.devPtr},
                            {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BNYAmax1.devPtr},
                            {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BNYAmax2.devPtr}})
        .setBNInScaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_scale0.devPtr},
                              {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_scale1.devPtr},
                              {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_scale2.devPtr}})
        .setBNInBiasDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_bias0.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_bias1.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_bias2.devPtr}})
        .setBNSavedMeanDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_saved_mean0.devPtr},
                                {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_saved_mean1.devPtr},
                                {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_saved_mean2.devPtr}})
        .setBNSavedInvVarDevPtrs(
            {{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_saved_inv_var0.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_saved_inv_var1.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_saved_inv_var2.devPtr}})
        .setBNRunningMeanDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_running_mean0.devPtr},
                                  {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_running_mean1.devPtr},
                                  {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_running_mean2.devPtr}})
        .setBNRunningVarDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_running_var0.devPtr},
                                 {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_running_var1.devPtr},
                                 {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_running_var2.devPtr}})
        .setXDescaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, XDescale0.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, XDescale1.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, XDescale2.devPtr}})
        .setWDescaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, WDescale0.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, WDescale1.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, WDescale2.devPtr}})
        .setYScaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, YScale0.devPtr},
                           {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, YScale1.devPtr},
                           {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, YScale2.devPtr}})
        .setYAmaxDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, YAmax0.devPtr},
                          {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, YAmax1.devPtr},
                          {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, YAmax2.devPtr}})
        .setBNZDeScale(ZDescale.devPtr)
        .setBNEpsilons({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, epsilons[0]},
                        {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, epsilons[1]},
                        {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, epsilons[2]}})
        .setBNExponentialAverageFactors(
            {{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, exp_avg_factors[0]},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, exp_avg_factors[1]},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, exp_avg_factors[2]}});

    cudnnStatus_t devPtrStoreStatus = devPtrStore.finalize();
    if (devPtrStoreStatus != CUDNN_STATUS_SUCCESS) {
        std::cout << devPtrStore.getErrorMessage() << std::endl;
        CHECK(false);
    }

    RunResidualBlock(residualBlockParams, &devPtrStore, "forward");
    RunResidualBlock(residualBlockParams, &devPtrStore, "forward_inference");
}

TEST_CASE("Residual block with residual conv", "[Resnet][Residual]") {
    int64_t const N = 204;
    int64_t const C = 64;
    int64_t const H = 56;
    int64_t const W = 56;

    int64_t const K[]      = {64, 64, 256, 256};
    int64_t const R[]      = {1, 3, 1, 1};
    int64_t const S[]      = {1, 3, 1, 1};
    int64_t const pad[]    = {0, 1, 0, 0};
    int64_t const stride[] = {1, 1, 1, 1};

    using fp8_e4m3 = int8_t;

    /* Start of device Pointer Creation */

    // Forward
    Surface<fp8_e4m3> X(N * C * H * W, false);

    Surface<float> XDescale0(1, false);
    Surface<float> XDescale1(1, false);
    Surface<float> XDescale2(1, false);
    // Residual's input descale is the same as first conv's input descale

    Surface<fp8_e4m3> WeightNHWC0(K[0] * C * R[0] * S[0], false);
    Surface<fp8_e4m3> WeightNHWC1(K[1] * K[0] * R[1] * S[1], false);
    Surface<fp8_e4m3> WeightNHWC2(K[2] * K[1] * R[2] * S[2], false);
    Surface<fp8_e4m3> WeightNHWC3(K[3] * C * R[3] * S[3], false);

    Surface<float> WDescale0(1, false);
    Surface<float> WDescale1(1, false);
    Surface<float> WDescale2(1, false);
    Surface<float> WDescale3(1, false);

    Surface<fp8_e4m3> Y0(N * K[0] * H * W, false);
    Surface<fp8_e4m3> Y1(N * K[1] * H * W, false);
    Surface<fp8_e4m3> Y2(N * K[2] * H * W, false);
    Surface<fp8_e4m3> Y3(N * K[3] * H * W, false);

    Surface<float> YScale0(1, false);
    Surface<float> YScale1(1, false);
    Surface<float> YScale2(1, false);
    Surface<float> YScale3(1, false);

    Surface<float> ZDescale(1, false);

    Surface<float> YAmax0(1, false);
    Surface<float> YAmax1(1, false);
    Surface<float> YAmax2(1, false);
    Surface<float> YAmax3(1, false);

    Surface<float> BNXDescale0(1, false);
    Surface<float> BNXDescale1(1, false);
    Surface<float> BNXDescale2(1, false);
    Surface<float> BNXDescale3(1, false);

    Surface<fp8_e4m3> BNY0(N * K[0] * H * W, false);
    Surface<fp8_e4m3> BNY1(N * K[1] * H * W, false);
    Surface<fp8_e4m3> BNY2(N * K[2] * H * W, false);
    Surface<fp8_e4m3> BNY3(N * K[3] * H * W, false);
    // BNY3 has been moved to workspace

    Surface<float> BNYAmax0(1, false);
    Surface<float> BNYAmax1(1, false);
    Surface<float> BNYAmax2(1, false);
    Surface<float> BNYAmax3(1, false);
    Surface<float> BNYScale0(1, false);
    Surface<float> BNYScale1(1, false);
    Surface<float> BNYScale2(1, false);
    Surface<float> BNYScale3(1, false);

    Surface<float> BN_running_mean0(K[0], false);
    Surface<float> BN_running_mean1(K[1], false);
    Surface<float> BN_running_mean2(K[2], false);
    Surface<float> BN_running_mean3(K[3], false);

    Surface<float> BN_running_var0(K[0], false);
    Surface<float> BN_running_var1(K[1], false);
    Surface<float> BN_running_var2(K[2], false);
    Surface<float> BN_running_var3(K[3], false);

    Surface<float> BN_saved_mean0(K[0], false);
    Surface<float> BN_saved_mean1(K[1], false);
    Surface<float> BN_saved_mean2(K[2], false);
    Surface<float> BN_saved_mean3(K[3], false);

    Surface<float> BN_saved_inv_var0(K[0], false);
    Surface<float> BN_saved_inv_var1(K[1], false);
    Surface<float> BN_saved_inv_var2(K[2], false);
    Surface<float> BN_saved_inv_var3(K[3], false);

    Surface<float> BN_scale0(K[0], false);
    Surface<float> BN_scale1(K[1], false);
    Surface<float> BN_scale2(K[2], false);
    Surface<float> BN_scale3(K[3], false);

    Surface<float> BN_bias0(K[0], false);
    Surface<float> BN_bias1(K[1], false);
    Surface<float> BN_bias2(K[2], false);
    Surface<float> BN_bias3(K[3], false);

    std::vector<float> epsilons        = {0.05f, 0.05f, 0.05f, 0.05f};
    std::vector<float> exp_avg_factors = {0.9f, 0.9f, 0.9f, 0.9f};

    //////////////////////////////////////////////////////////////////////////////////

    /* End of device Pointer Creation */

    // Set the convolution parameters
    auto residualBlockParams = cudnn_frontend::ResidualBlockParamsBuilder()
                                   .setInputDim({N, C, H, W})
                                   .setConvolutionParams(cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO,
                                                         {K[0], R[0], S[0], pad[0], stride[0]})
                                   .setConvolutionParams(cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE,
                                                         {K[1], R[1], S[1], pad[1], stride[1]})
                                   .setConvolutionParams(cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO,
                                                         {K[2], R[2], S[2], pad[2], stride[2]})
                                   .setConvolutionParams(cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL,
                                                         {K[3], R[3], S[3], pad[3], stride[3]})
                                   .build();

    ResidualBlockDevPtrStore devPtrStore;

    // Set device ptrs
    devPtrStore.setXDevPtr(X.devPtr)
        .setYDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, Y0.devPtr},
                      {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, Y1.devPtr},
                      {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, Y2.devPtr},
                      {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, Y3.devPtr}})
        .setWeightNHWCDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, WeightNHWC0.devPtr},
                               {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, WeightNHWC1.devPtr},
                               {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, WeightNHWC2.devPtr},
                               {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, WeightNHWC3.devPtr}})
        .setBNXDescaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BNXDescale0.devPtr},
                               {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BNXDescale1.devPtr},
                               {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BNXDescale2.devPtr},
                               {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, BNXDescale3.devPtr}})
        .setBNYDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BNY0.devPtr},
                        {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BNY1.devPtr},
                        {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BNY2.devPtr},
                        {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, BNY3.devPtr}})
        .setBNYScaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BNYScale0.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BNYScale1.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BNYScale2.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, BNYScale3.devPtr}})
        .setBNYAMaxDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BNYAmax0.devPtr},
                            {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BNYAmax1.devPtr},
                            {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BNYAmax2.devPtr},
                            {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, BNYAmax3.devPtr}})
        .setBNInScaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_scale0.devPtr},
                              {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_scale1.devPtr},
                              {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_scale2.devPtr},
                              {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, BN_scale3.devPtr}})
        .setBNInBiasDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_bias0.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_bias1.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_bias2.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, BN_bias3.devPtr}})
        .setBNSavedMeanDevPtrs(
            {{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_saved_mean0.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_saved_mean1.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_saved_mean2.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, BN_saved_mean3.devPtr}})
        .setBNSavedInvVarDevPtrs(
            {{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_saved_inv_var0.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_saved_inv_var1.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_saved_inv_var2.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, BN_saved_inv_var3.devPtr}})
        .setBNRunningMeanDevPtrs(
            {{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_running_mean0.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_running_mean1.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_running_mean2.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, BN_running_mean3.devPtr}})
        .setBNRunningVarDevPtrs(
            {{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, BN_running_var0.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, BN_running_var1.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, BN_running_var2.devPtr},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, BN_running_var3.devPtr}})
        .setXDescaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, XDescale0.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, XDescale1.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, XDescale2.devPtr}})
        .setWDescaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, WDescale0.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, WDescale1.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, WDescale2.devPtr},
                             {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, WDescale3.devPtr}})
        .setYScaleDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, YScale0.devPtr},
                           {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, YScale1.devPtr},
                           {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, YScale2.devPtr},
                           {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, YScale3.devPtr}})
        .setYAmaxDevPtrs({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, YAmax0.devPtr},
                          {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, YAmax1.devPtr},
                          {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, YAmax2.devPtr},
                          {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, YAmax3.devPtr}})
        .setBNZDeScale(ZDescale.devPtr)
        .setBNEpsilons({{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, epsilons[0]},
                        {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, epsilons[1]},
                        {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, epsilons[2]},
                        {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, epsilons[3]}})
        .setBNExponentialAverageFactors(
            {{cudnn_frontend::ResidualBlockParams::ForwardLocation::ZERO, exp_avg_factors[0]},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::ONE, exp_avg_factors[1]},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::TWO, exp_avg_factors[2]},
             {cudnn_frontend::ResidualBlockParams::ForwardLocation::RESIDUAL, exp_avg_factors[3]}});

    cudnnStatus_t devPtrStoreStatus = devPtrStore.finalize();
    if (devPtrStoreStatus != CUDNN_STATUS_SUCCESS) {
        std::cout << devPtrStore.getErrorMessage() << std::endl;
        CHECK(false);
    }

    RunResidualBlock(residualBlockParams, &devPtrStore, "forward");
    RunResidualBlock(residualBlockParams, &devPtrStore, "forward_inference");
}

#endif
