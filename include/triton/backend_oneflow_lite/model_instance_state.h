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
#include <memory>
#include <string>

#include "model_state.h"

#include "oneflow-lite/core/executable.h"
#include "oneflow-lite/core/execution_context.h"

#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace oneflow_lite {

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  TRITONSERVER_Error* LoadModel();
  void ProcessRequests(
      TRITONBACKEND_Request** requests, uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  void Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count);
  void SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  void ReadOutputTensors(
      size_t total_batch_size, const std::vector<std::string>& output_names,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  bool CountBatchSize(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      size_t* total_batch_size);

  ModelState* model_state_;

  OfLiteExecutable* executable_;
  OfLiteExecutionContext* context_;
};


}}}  // namespace triton::backend::oneflow_lite
