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

#include <oneflow/tensor.h>
#include <cstddef>
#include <iostream>
#include <string>

namespace triton { namespace backend { namespace oneflow {

enum class XrtKind : int { kOneflow = 0, kTensorrt = 1, kOpenvino = 2 };

inline XrtKind
ParseXrtKind(const std::string& xrt_str, bool* is_unknown)
{
  XrtKind kind = XrtKind::kOneflow;
  *is_unknown = true;
  if (xrt_str == "oneflow") {
    kind = XrtKind::kOneflow;
    *is_unknown = false;
  }
  else if (xrt_str == "tensorrt") {
    kind = XrtKind::kTensorrt;
    *is_unknown = false;
  }
  else if (xrt_str == "openvino") {
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
GetDataPtrFp32(const oneflow_api::Tensor& tensor, float** dptr, size_t* elem_cnt) {
  *elem_cnt = tensor.shape().elem_cnt();
  *dptr = new float[*elem_cnt * 4];
  oneflow_api::Tensor::to_blob(tensor, *dptr);
  return true;
}

inline void
PrintTensor(const oneflow_api::Tensor& tensor) {
  float* dptr;
  size_t elem_cnt;
  GetDataPtrFp32(tensor, &dptr, &elem_cnt);
  for (size_t i = 0; i < elem_cnt; i++) {
    std::cout << dptr[i] << " ";
  }
  std::cout << std::endl;
  delete [] dptr;
}

}}}  // namespace triton::backend::oneflow
