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

#include <fstream>
#include <iostream>
#include <sstream>
#include <variant>
#include <vector>

#include "../../cuda/cuda_common.h"

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                      \
  {                                                                \
    cutlass::Status error = status;                                \
    CHECK(error == cutlass::Status::kSuccess)                      \
        << "Got cutlass error: " << cutlassGetStatusString(error); \
  }

using namespace cute;
using ProblemShape = Shape<int, int, int, int>;

template <typename KernelTraits, typename ElementC, typename LayoutA = cutlass::layout::RowMajor,
          typename LayoutB = cutlass::layout::ColumnMajor,
          typename LayoutC = cutlass::layout::RowMajor>
struct CutlassGemmRunner {
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;

  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Alignment of C matrix in units of elements

  // Core kernel configurations
  using ElementAccumulator = float;  // Element type for internal accumulation
  using ScaleType = std::variant<ElementAccumulator, const ElementAccumulator*>;
  using ArchTag =
      cutlass::arch::Sm100;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using MmaTileShape = typename KernelTraits::MmaTileShape;
  using ClusterShape = typename KernelTraits::ClusterShape;
  using PerSmTileShape_MNK = typename KernelTraits::PerSmTileShape_MNK;
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based on the tile size
  using KernelSchedule = typename KernelTraits::KernelSchedule;  // Kernel to launch
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;  // Epilogue to launch

  using Element = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC, EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, Element, LayoutA, AlignmentA, Element, LayoutB, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                                          CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;

  void run_gemm(const ElementA* ptr_A, const ElementB* ptr_B, const ElementC* ptr_C,
                ElementC* ptr_D, ProblemShape* problem_size, StrideA* stride_A, StrideB* stride_B,
                StrideC* stride_C, StrideD* stride_D, uint8_t* workspace, int64_t workspace_size,
                ScaleType alpha, ScaleType beta, cudaStream_t stream, int64_t M, int64_t N,
                int64_t K, ElementSFA* ptr_SFA, ElementSFB* ptr_SFB) {
    int m = static_cast<int>(M);
    int n = static_cast<int>(N);
    int k = static_cast<int>(K);
    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    using Sm100BlkScaledConfig =
        typename Gemm::GemmKernel::CollectiveMainloop::Sm100BlkScaledConfig;
    auto layout_SFA = Sm100BlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_SFB = Sm100BlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {// Mainloop arguments
         ptr_A, stride_a, ptr_B, stride_b, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB},
        {     // Epilogue arguments
         {},  // alpha, beta
         ptr_C,
         stride_c,
         ptr_D,
         stride_d}};
    ICHECK(alpha.index() == beta.index()) << "alpha and beta must have the same type";
    if (std::holds_alternative<ElementAccumulator>(alpha)) {
      arguments.epilogue.thread.alpha = std::get<ElementAccumulator>(alpha);
      arguments.epilogue.thread.beta = std::get<ElementAccumulator>(beta);
    } else if (std::holds_alternative<const ElementAccumulator*>(alpha)) {
      arguments.epilogue.thread.alpha_ptr = std::get<const ElementAccumulator*>(alpha);
      arguments.epilogue.thread.beta_ptr = std::get<const ElementAccumulator*>(beta);
    } else {
      LOG(FATAL) << "Unsupported alpha and beta type";
      throw;
    }

    Gemm gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CHECK_GE(workspace_size, gemm_op.get_workspace_size(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace, stream));
    CUTLASS_CHECK(
        gemm_op.run(arguments, workspace, stream));  // gemm_op.run(stream); gemm_op.run(stream)
  }
};

template <typename KernelTraits, typename ElementC>
void cutlass_gemm_fp4(cutlass::float_e2m1_t* x, cutlass::float_e2m1_t* weight, uint8_t* workspace,
                      int64_t workspace_size, int64_t m, int64_t n, int64_t k,
                      std::variant<float, const float*> alpha,
                      std::variant<float, const float*> beta, ElementC* out, cudaStream_t stream,
                      cutlass::float_ue4m3_t* data_sfa, cutlass::float_ue4m3_t* data_sfb) {
  // Use the ElementA and ElementB types defined within the Runner
  using Runner = CutlassGemmRunner<KernelTraits, ElementC>;
  using InternalElementA = typename Runner::ElementA;
  using InternalElementB = typename Runner::ElementB;
  using StrideA = typename Runner::StrideA;
  using StrideB = typename Runner::StrideB;
  using StrideC = typename Runner::StrideC;  // Assuming D uses StrideC layout

  Runner runner;
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  ProblemShape problem_size{static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1};
  // Cast pointers to the type expected by the runner's run_gemm function
  auto ptr_A = reinterpret_cast<const InternalElementA*>(x);
  auto ptr_B = reinterpret_cast<const InternalElementB*>(weight);
  runner.run_gemm(x, weight, out, out, &problem_size, &stride_A, &stride_B, &stride_D, &stride_D,
                  workspace, workspace_size, alpha, beta, stream, m, n, k, data_sfa, data_sfb);
}
