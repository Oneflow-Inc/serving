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

#include <oneflow/device.h>
#include <oneflow/dtype.h>
#include <oneflow/env.h>
#include <oneflow/graph.h>
#include <oneflow/shape.h>
#include <oneflow/tensor.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <ostream>
#include <thread>
#include <vector>

#include "oneflow/api.h"
#include "oneflow_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace backend { namespace oneflow {

//
// Simple backend that demonstrates the TRITONBACKEND API for a blocking
// backend. A blocking backend completes execution of the inference before
// returning from TRITONBACKEND_ModelInstanceExecute.
//
// This backend supports any model that has same number of inputs and outputs.
// The input and output must follow a naming convention i.e INPUT<index> and
// OUTPUT<index> where 'index' is the input/output index. The datatype and
// shape/reshaped shape must match. The backend simply responds with the output
// tensors equal to the corresponding input tensors.
//

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
  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  XrtKind xrt_kind;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), xrt_kind(XrtKind::kOneflow)
{
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // Collect input/output names, shapes and datatypes
  std::map<std::string, std::tuple<std::string, std::vector<int64_t>>>
      input_infos, output_infos;
  for (size_t io_index = 0; io_index < inputs.ArraySize(); io_index++) {
    common::TritonJson::Value input, output;
    RETURN_IF_ERROR(inputs.IndexAsObject(io_index, &input));
    RETURN_IF_ERROR(outputs.IndexAsObject(io_index, &output));

    const char* input_name;
    size_t input_name_len;
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));

    const char* output_name;
    size_t output_name_len;
    RETURN_IF_ERROR(
        output.MemberAsString("name", &output_name, &output_name_len));

    std::string input_name_str = std::string(input_name);
    std::string output_name_str = std::string(output_name);

    // Input and output must have same datatype
    std::string input_dtype, output_dtype;
    RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
    RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));

    // Input and output must have same shape or reshaped shape
    std::vector<int64_t> input_shape, output_shape;
    triton::common::TritonJson::Value reshape;
    if (input.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(backend::ParseShape(reshape, "shape", &input_shape));
    } else {
      RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
    }

    if (output.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(backend::ParseShape(reshape, "shape", &output_shape));
    } else {
      RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));
    }

    input_infos.insert(std::make_pair(
        input_name_str, std::make_tuple(input_dtype, input_shape)));
    output_infos.insert(std::make_pair(
        output_name_str, std::make_tuple(output_dtype, output_shape)));
  }

  triton::common::TritonJson::Value params;
  bool is_unknown = true;
  if (model_config_.Find("parameters", &params)) {
    common::TritonJson::Value xrt;
    if (params.Find("xrt", &xrt)) {
      std::string xrt_str;
      RETURN_IF_ERROR(xrt.MemberAsString("string_value", &xrt_str));
      this->xrt_kind = ParseXrtKind(xrt_str, &is_unknown);
    }
  }
  if (is_unknown) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        "xrt tag is unknown, oneflow runtime will be used");
  }

  return nullptr;  // success
}

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
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  ModelState* model_state_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(
        model_state, triton_model_instance);
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
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
}

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  oneflow_api::initialize();
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  oneflow_api::release();
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: ") + name)
          .c_str());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  std::string device_tag = "cpu";
#ifdef TRITON_ENABLE_GPU
  if (kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    // cudaSetDevice(device_id);
    device_tag = "gpu";
  }
#endif  // TRITON_ENABLE_GPU

  oneflow_api::Device device(device_tag, device_id);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

#ifdef TRITON_ENABLE_GPU
  if (instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaSetDevice(instance_state->DeviceId());
  }
#endif  // TRITON_ENABLE_GPU

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

#ifdef TRITON_ENABLE_GPU
  if (instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaSetDevice(instance_state->DeviceId());
  }
#endif  // TRITON_ENABLE_GPU

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  bool supports_batching = false;
  RETURN_IF_ERROR(model_state->SupportsFirstDimBatching(&supports_batching));

  // 'responses' is initialized with the response objects below and
  // if/when an error response is sent the corresponding entry in
  // 'responses' is set to nullptr to indicate that that response has
  // already been sent.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  // Create a single response object for each request. If something
  // goes wrong when attempting to create the response objects just
  // fail all of the requests by returning an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  // The way we collect these batch timestamps is not entirely
  // accurate. Normally, in a performant backend you would execute all
  // the requests at the same time, and so there would be a single
  // compute-start / compute-end time-range. But here we execute each
  // request separately so there is no single range. As a result we
  // just show the entire execute time as being the compute time as
  // well.

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Delay if requested...
  // if (model_state->ExecDelay() > 0) {
  //   uint64_t multiplier = model_state->DelayMultiplier();

  //   if (model_state->DelayMultiplier() > 0) {
  //     multiplier *= instance_state->InstanceId();
  //   }
  //   uint64_t delay_ms =
  //       model_state->ExecDelay() * std::max(multiplier, uint64_t(1));
  //   std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  // }

  // For simplicity we just process each request separately... in
  // general a backend should try to operate on the entire batch of
  // requests at the same time for improved performance.
  uint64_t total_batch_size = 0;
  bool cuda_copy = false;
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &request_id));

    uint64_t correlation_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

    uint32_t input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInputCount(request, &input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    // If an error response was sent for the above then display an error message
    // and move on to next request.
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read request input/output counts, error response sent")
              .c_str());
      continue;
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("request ") + std::to_string(r) + ": id = \"" +
         request_id + "\", correlation_id = " + std::to_string(correlation_id) +
         ", input_count = " + std::to_string(input_count) +
         ", requested_output_count = " + std::to_string(requested_output_count))
            .c_str());

    // Collect all input indices
    std::set<int32_t> input_indices;
    for (uint32_t io_index = 0; io_index < input_count; io_index++) {
      const char* input_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestInputName(request, io_index, &input_name));
      input_indices.insert(std::stoi(std::string(input_name, 5, -1)));
    }

    // For statistics we need to collect the total batch size of all the
    // requests. If the model doesn't support batching then each request is
    // necessarily batch-size 1. If the model does support batching then the
    // first dimension of the shape is the batch size. We only the first input
    // for this.
    if (supports_batching) {
      TRITONBACKEND_Input* input = nullptr;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to read input, error response sent")
                .c_str());
        continue;
      }

      const int64_t* input_shape;
      uint32_t input_dims_count;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_InputProperties(
              input, nullptr, nullptr, &input_shape, &input_dims_count, nullptr,
              nullptr));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to read input properties, error response sent")
                .c_str());
        continue;
      }

      if (input_dims_count > 0) {
        total_batch_size += input_shape[0];
      } else {
        total_batch_size++;
      }
    }

    // We validated that the model configuration specifies N inputs, but the
    // request is not required to request any output at all so we only produce
    // outputs that are requested.
    for (uint32_t io_index = 0; io_index < requested_output_count; io_index++) {
      const char* output_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(request, io_index, &output_name));

      // If an error response was sent while getting the requested output name
      // then display an error message and move on to next request.
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to read requested output name, error response sent")
                .c_str());
        continue;
      }

      std::string index_str = std::string(output_name, 6, -1);
      std::string input_name = "INPUT" + index_str;
      TRITONBACKEND_Input* input = nullptr;
      if (input_indices.find(std::stoi(index_str)) != input_indices.end()) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_RequestInput(request, input_name.c_str(), &input));

        // If an error response was sent while getting the input then display an
        // error message and move on to next request.
        if (responses[r] == nullptr) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to read input, error response sent")
                  .c_str());
          continue;
        }
      } else {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                ("failed to get input '" + input_name + "'").c_str()));
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to get input '" + input_name + "', error response sent")
                .c_str());
        continue;
      }

      TRITONSERVER_DataType input_datatype;
      const int64_t* input_shape;
      uint32_t input_dims_count;
      uint64_t input_byte_size;
      uint32_t input_buffer_count;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_InputProperties(
              input, nullptr /* input_name */, &input_datatype, &input_shape,
              &input_dims_count, &input_byte_size, &input_buffer_count));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to read input properties, error response sent")
                .c_str());
        continue;
      }

      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("\tinput ") + input_name + ": datatype = " +
           TRITONSERVER_DataTypeString(input_datatype) + ", shape = " +
           backend::ShapeToString(input_shape, input_dims_count) +
           ", byte_size = " + std::to_string(input_byte_size) +
           ", buffer_count = " + std::to_string(input_buffer_count))
              .c_str());

      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("\trequested_output ") + output_name).c_str());

      // This backend simply copies the output tensors from the corresponding
      // input tensors. The input tensors contents are available in one or more
      // contiguous buffers. To do the copy we:
      //
      //   1. Create an output tensor in the response.
      //
      //   2. Allocate appropriately sized buffer in the output
      //      tensor.
      //
      //   3. Iterate over the input tensor buffers and copy the contents into
      //      the output buffer.
      TRITONBACKEND_Response* response = responses[r];

      // Step 1. Create an output tensor. Input and output have same datatype
      // and shape/reshaped shape
      TRITONBACKEND_Output* output;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &output, output_name, input_datatype, input_shape,
              input_dims_count));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to create response output, error response sent")
                .c_str());
        continue;
      }

      // Step 2. Get the output buffer.
      void* output_buffer;
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, input_byte_size, &output_memory_type,
              &output_memory_type_id));

      if (responses[r] == nullptr) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to create output buffer"));
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to create output buffer, error response sent")
                .c_str());
        continue;
      }

      // Step 3. Copy input -> output
      size_t output_buffer_offset = 0;
      for (uint32_t b = 0; b < input_buffer_count; ++b) {
        const void* input_buffer = nullptr;
        uint64_t buffer_byte_size = 0;
        TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t input_memory_type_id = 0;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputBuffer(
                input, b, &input_buffer, &buffer_byte_size, &input_memory_type,
                &input_memory_type_id));

        if (responses[r] == nullptr) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "failed to get input buffer"));
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to get input buffer, error response sent")
                  .c_str());
          continue;
        }

        // forward here
        std::vector<int64_t> shape_vec;
        for (uint32_t shape_index = 0; shape_index < input_dims_count; shape_index++) {
          shape_vec.push_back(input_shape[shape_index]);
        }
        oneflow_api::Shape shape = oneflow_api::Shape(shape_vec);
        oneflow_api::Device device("cpu");
        oneflow_api::DType dtype = oneflow_api::DType::kFloat;
        oneflow_api::Tensor tensor = oneflow_api::Tensor::from_blob(input_buffer, shape, device, dtype);
        tensor = oneflow_api::nn::relu(tensor);
        std::array<float, 10> relu_data;
        oneflow_api::Tensor::to_blob(tensor, relu_data.data());
        std::cout << "result: ";
        for (float ele : relu_data) {
            std::cout << ele << " ";
        }
        std::cout << std::endl;

        bool cuda_used = false;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            CopyBuffer(
                input_name, input_memory_type, input_memory_type_id,
                output_memory_type, output_memory_type_id, buffer_byte_size,
                relu_data.data(),
                reinterpret_cast<char*>(output_buffer) + output_buffer_offset,
                instance_state->CudaStream(), &cuda_used));
        cuda_copy |= cuda_used;

        if (responses[r] == nullptr) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED, "failed to get copy buffer"));
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to get copy buffer, error response sent")
                  .c_str());
          continue;
        }
        output_buffer_offset += buffer_byte_size;
      }
    }

    // To demonstrate response parameters we attach some here. Most responses do
    // not use parameters but they provide a way for backends to communicate
    // arbitrary information along with the response.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetStringParameter(
            responses[r], "param0", "an example string parameter"),
        "failed setting string parameter");
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetIntParameter(responses[r], "param1", 42),
        "failed setting integer parameter");
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetBoolParameter(responses[r], "param2", false),
        "failed setting boolean parameter");
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(instance_state->CudaStream());
  }
#endif  // TRITON_ENABLE_GPU

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // If we get to this point then there hasn't been any error and the response
  // is complete and we can send it. This is the last (and only) response that
  // we are sending for the request so we must mark it FINAL. If there is an
  // error when sending all we can do is log it.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed sending response");
    }
  }

  // There are two types of statistics that we can report... the statistics for
  // the entire batch of requests that we just executed and statistics for each
  // individual request. Here we report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Here we report statistics for the entire batch of requests.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), total_batch_size,
          exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::oneflow
