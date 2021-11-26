// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <oneflow/dtype.h>
#include <oneflow/tensor.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>

#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace oneflow {

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)


#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return;                                                          \
    }                                                                  \
  } while (false)

enum class XrtKind : int { kOneflow = 0, kTensorrt = 1, kOpenvino = 2 };

inline XrtKind
ParseXrtKind(const std::string& xrt_str, bool* is_unknown)
{
  XrtKind kind = XrtKind::kOneflow;
  *is_unknown = true;
  if (xrt_str == "oneflow") {
    kind = XrtKind::kOneflow;
    *is_unknown = false;
  } else if (xrt_str == "tensorrt") {
    kind = XrtKind::kTensorrt;
    *is_unknown = false;
  } else if (xrt_str == "openvino") {
    kind = XrtKind::kOpenvino;
    *is_unknown = false;
  }
  return kind;
}

inline bool
IsXrtOneFlow(const XrtKind& kind)
{
  return kind == XrtKind::kOneflow;
}

inline bool
IsXrtTensorrt(const XrtKind& kind)
{
  return kind == XrtKind::kTensorrt;
}

inline bool
IsXrtOpenvino(const XrtKind& kind)
{
  return kind == XrtKind::kOpenvino;
}

// TODO(zzk0): delete this or refactor; used to test currently
inline bool
GetDataPtrFp32(
    const oneflow_api::Tensor& tensor, float** dptr, size_t* elem_cnt)
{
  *elem_cnt = tensor.shape().elem_cnt();
  *dptr = new float[*elem_cnt * 4];
  oneflow_api::Tensor::to_blob(tensor, *dptr);
  return true;
}

inline void
PrintTensor(const oneflow_api::Tensor& tensor)
{
  float* dptr;
  size_t elem_cnt;
  GetDataPtrFp32(tensor, &dptr, &elem_cnt);
  for (size_t i = 0; i < elem_cnt; i++) {
    std::cout << dptr[i] << " ";
  }
  std::cout << std::endl;
  delete[] dptr;
}

inline TRITONSERVER_DataType
ConvertOneFlowTypeToTritonType(const oneflow_api::DType& of_type)
{
  switch (of_type) {
    case oneflow_api::DType::kFloat:
      return TRITONSERVER_TYPE_FP32;
    case oneflow_api::DType::kDouble:
      return TRITONSERVER_TYPE_FP64;
    case oneflow_api::DType::kInt8:
      return TRITONSERVER_TYPE_INT8;
    case oneflow_api::DType::kInt32:
      return TRITONSERVER_TYPE_INT32;
    case oneflow_api::DType::kInt64:
      return TRITONSERVER_TYPE_INT64;
    case oneflow_api::DType::kUInt8:
      return TRITONSERVER_TYPE_UINT8;
    case oneflow_api::DType::kFloat16:
      return TRITONSERVER_TYPE_FP16;
    case oneflow_api::DType::kChar:
    case oneflow_api::DType::kOFRecord:
    case oneflow_api::DType::kTensorBuffer:
    case oneflow_api::DType::kBFloat16:
    case oneflow_api::DType::kMaxDataType:
    case oneflow_api::DType::kInvalidDataType:
    default:
      return TRITONSERVER_TYPE_INVALID;
  }
}

inline oneflow_api::DType
ConvertTritonTypeToOneFlowType(const TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_UINT8:
      return oneflow_api::DType::kUInt8;
    case TRITONSERVER_TYPE_INT8:
      return oneflow_api::DType::kInt8;
    case TRITONSERVER_TYPE_INT32:
      return oneflow_api::DType::kInt32;
    case TRITONSERVER_TYPE_INT64:
      return oneflow_api::DType::kInt64;
    case TRITONSERVER_TYPE_FP16:
      return oneflow_api::DType::kFloat16;
    case TRITONSERVER_TYPE_FP32:
      return oneflow_api::DType::kFloat;
    case TRITONSERVER_TYPE_FP64:
      return oneflow_api::DType::kDouble;
    case TRITONSERVER_TYPE_INVALID:
    case TRITONSERVER_TYPE_BOOL:
    case TRITONSERVER_TYPE_INT16:
    case TRITONSERVER_TYPE_BYTES:
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_UINT64:
    default:
      return oneflow_api::DType::kInvalidDataType;
  }
}

}}}  // namespace triton::backend::oneflow
