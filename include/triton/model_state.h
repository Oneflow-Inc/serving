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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "oneflow/api.h"
#include "oneflow_utils.h"
#include "triton/backend/backend_model.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace oneflow {

struct InputOutputAttribute {
  TRITONSERVER_DataType datatype_;
  std::vector<int64_t> input_output_shape_;
  size_t input_output_index_;
};

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateAndParseModelConfig();
  const std::vector<std::string>& InputNames() const;
  const std::vector<std::string>& OutputNames() const;
  const std::unordered_map<std::string, InputOutputAttribute>& InputAttributes()
      const;
  const std::unordered_map<std::string, InputOutputAttribute>&
  OutputAttributes() const;

  TRITONSERVER_Error* LoadModel(
      const oneflow_api::Device device,
      std::unique_ptr<oneflow_api::Graph>* graph);

 private:
  TRITONSERVER_Error* AutoCompleteConfig();
  TRITONSERVER_Error* AutoCompleteInputsAndOutputs(
      const char* key, oneflow_api::InputOutputInfos& input_output_infos);
  TRITONSERVER_Error* AutoCompleteMaxBatchSize();

  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* ValidateAndParseInputs();
  TRITONSERVER_Error* ValidateAndParseOutputs();

  XrtKind xrt_kind_ = XrtKind::kOneflow;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, InputOutputAttribute> input_attribute_;
  std::unordered_map<std::string, InputOutputAttribute> output_attribute_;
};

}}}  // namespace triton::backend::oneflow
