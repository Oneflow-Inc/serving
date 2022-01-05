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

#include "model_instance_state.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "oneflow/api.h"
#include "oneflow_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace oneflow {

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state, const oneflow_api::Device& device)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance, device);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const oneflow_api::Device& device)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), device_(device)
{
}

TRITONSERVER_Error*
ModelInstanceState::LoadModel()
{
  RETURN_IF_ERROR(model_state_->LoadModel(device_, &graph_));
  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  size_t total_batch_size = 0;
  if (!CountBatchSize(requests, request_count, &total_batch_size)) {
    return;
  }

  // create responses
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  for (size_t i = 0; i < request_count; ++i) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // collect input
  std::vector<const char*> input_names;
  std::vector<oneflow_api::Tensor> input_tensors;
  std::vector<BackendMemory*> input_memories;
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream());
  SetInputTensors(
      total_batch_size, requests, request_count, &responses, &collector,
      &input_names, &input_tensors, &input_memories, &cuda_copy);
  SynchronizeStream(CudaStream(), cuda_copy);

  // execute
  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  std::vector<oneflow_api::Tensor> output_tensors;
  Execute(&responses, request_count, &input_tensors, &output_tensors);

  for (BackendMemory* mem : input_memories) {
    delete mem;
  }
  input_memories.clear();

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);
  cuda_copy = false;
  ReadOutputTensors(
      total_batch_size, model_state_->OutputNames(), output_tensors, requests,
      request_count, &responses, &cuda_copy);

  // report
  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send OneFlow backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
}

void
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<oneflow_api::Tensor>* input_tensors,
    std::vector<BackendMemory*>* input_memories, bool* cuda_copy)
{
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  input_tensors->resize(input_count);
  for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx) {
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            input, &input_name, &input_datatype, &input_shape,
            &input_dims_count, nullptr, nullptr));

    input_names->emplace_back(input_name);
    std::vector<int64_t> tensor_shape(
        input_shape, input_shape + input_dims_count);

    // padding to max batch size
    int max_batch_size = model_state_->MaxBatchSize();
    if (max_batch_size != 0) {
      tensor_shape[0] = max_batch_size;
    }
    const int64_t tensor_byte_size = GetByteSize(input_datatype, tensor_shape);

    std::vector<BackendMemory::AllocationType> alloc_perference;
    if (device_.type() == "cpu") {
      alloc_perference = {BackendMemory::AllocationType::CPU};
    } else {
      alloc_perference = {
          BackendMemory::AllocationType::GPU_POOL,
          BackendMemory::AllocationType::GPU};
    }

    BackendMemory* input_memory;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        BackendMemory::Create(
            model_state_->TritonMemoryManager(), alloc_perference,
            device_.type() == "cpu" ? 0 : device_.device_id(), tensor_byte_size,
            &input_memory));
    input_memories->push_back(input_memory);

    TRITONSERVER_MemoryType memory_type = input_memory->MemoryType();
    int64_t memory_type_id = input_memory->MemoryTypeId();
    char* input_buffer = input_memory->MemoryPtr();

    collector->ProcessTensor(
        input_name, input_buffer, tensor_byte_size, memory_type,
        memory_type_id);

    oneflow_api::DType of_type = ConvertTritonTypeToOneFlowType(input_datatype);
    oneflow_api::Shape shape(tensor_shape);
    oneflow_api::Tensor input_tensor = oneflow_api::Tensor::from_buffer(
        reinterpret_cast<float*>(input_buffer), shape, device_, of_type);

    auto input_attribute = model_state_->InputAttributes().find(input_name);
    if (input_attribute == model_state_->InputAttributes().end()) {
      continue;
    }
    size_t input_tensor_index = input_attribute->second.input_output_index;
    (*input_tensors)[input_tensor_index] = input_tensor;
  }

  *cuda_copy |= collector->Finalize();
}

void
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<std::string>& output_names,
    const std::vector<oneflow_api::Tensor>& output_tensors,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses, bool* cuda_copy)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->MaxBatchSize(),
      model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
      CudaStream());

  for (size_t idx = 0; idx < output_names.size(); ++idx) {
    const std::string& name = output_names[idx];
    auto output_attribute = model_state_->OutputAttributes().find(name);
    if (output_attribute == model_state_->OutputAttributes().end()) {
      continue;
    }
    size_t output_tensor_index = output_attribute->second.input_output_index;
    const oneflow_api::Tensor& output_tensor =
        output_tensors[output_tensor_index];
    TRITONSERVER_DataType output_dtype = output_attribute->second.datatype;
    std::vector<int64_t> tensor_shape = OfShapeToVector(output_tensor.shape());
    int64_t output_buffer_size = GetByteSize(output_dtype, tensor_shape);
    std::vector<char> output_buffer(output_buffer_size);
    output_tensor.copy_to(reinterpret_cast<float*>(output_buffer.data()));
    responder.ProcessTensor(
        name, output_dtype, tensor_shape, output_buffer.data(),
        TRITONSERVER_MEMORY_CPU, 0);

    // TODO(zzk0): avoid copy twice
    // responder.ProcessTensor(
    //     name, output_dtype, tensor_shape, output_buffer.data(),
    //     device_.type() == "cpu" ? TRITONSERVER_MEMORY_CPU :
    //     TRITONSERVER_MEMORY_GPU, device_.type() == "cpu" ? 0 :
    //     device_.device_id());
  }

  *cuda_copy |= responder.Finalize();
}

void
ModelInstanceState::Execute(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count,
    std::vector<oneflow_api::Tensor>* input_tensors,
    std::vector<oneflow_api::Tensor>* output_tensors)
{
  const auto& output = graph_->Forward(*input_tensors);
  if (output.IsTensor()) {
    output_tensors->push_back(output.ToTensor());
  } else {
    *output_tensors = output.ToTensorVector();
  }
}

bool
ModelInstanceState::CountBatchSize(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    size_t* total_batch_size)
{
  *total_batch_size = 0;
  const int max_batch_size = model_state_->MaxBatchSize();
  for (size_t i = 0; i < request_count; ++i) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to OneFlow backend for '" + Name() + "'")
                  .c_str()));
      return false;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        *total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return false;
      }
    } else {
      *total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (*total_batch_size == 0) {
    return false;
  }

  return true;
}

}}}  // namespace triton::backend::oneflow
