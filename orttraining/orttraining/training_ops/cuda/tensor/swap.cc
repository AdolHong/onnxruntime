// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/framework/memcpy.h"
#include "core/providers/cuda/gpu_data_transfer.h"

namespace onnxruntime {
namespace cuda {

// this is just MemcpyTo/FromHost with a different OpType for easier analysis

ONNX_OPERATOR_KERNEL_EX(
    SwapToHost,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .ExecQueueId(kCudaStreamCopyOut)
        .OutputMemoryType<OrtMemTypeCPUInput>(0),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    SwapFromHost,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .ExecQueueId(kCudaStreamCopyIn)
        .InputMemoryType<OrtMemTypeCPUInput>(0),
    Memcpy);

}  // namespace cuda
}  // namespace onnxruntime
