/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <cuda_fp16.h>
#include <float.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

// #include "../cublas/cublas_utils.h"
#include "fp4_gemm_runner.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

struct KernelTraits {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

namespace tvm {
namespace runtime {

template <typename ElementA, typename ElementB, typename ElementC>
void tvm_cutlass_fp4_gemm(NDArray x, NDArray weight, NDArray workspace, NDArray alpha,
                          NDArray out) {
  // Workspace is used for storing device-side gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  CHECK_GE(x->ndim, 2);
  CHECK_EQ(weight->ndim, 2);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_GE(out->ndim, 2);
  CHECK_EQ(alpha->dtype.code, kDLFloat);
  CHECK_EQ(alpha->dtype.bits, 32);
  CHECK_EQ(alpha->ndim, 1);
  CHECK_EQ(alpha->shape[0], 1);
  int64_t m = 1;
  for (int i = 0; i < x->ndim - 1; ++i) {
    m *= x->shape[i];
  }
  int64_t n = weight->shape[0];
  CHECK_EQ(x->shape[x->ndim - 1], weight->shape[1]) << "Only col-major weight is supported now.";
  int64_t k = x->shape[x->ndim - 1];
  const float* beta = nullptr;
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
  // if (m <= 64) {
  cutlass_gemm<KernelTraits>(
      static_cast<ElementA*>(x->data), static_cast<ElementB*>(weight->data),
      static_cast<uint8_t*>(workspace->data), workspace->shape[0], m, n, k,
      static_cast<float*>(alpha->data), beta, static_cast<ElementC*>(out->data), stream);

}


TVM_REGISTER_GLOBAL("cutlass.gemm_e2m1_e2m1_fp16")
    .set_body_typed(
        tvm_cutlass_fp4_gemm<cutlass::float_e2m1_t, cutlass::float_e2m1_t, cutlass::half_t>);

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
