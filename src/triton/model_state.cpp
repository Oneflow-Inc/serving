#include "model_state.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "oneflow_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace oneflow {

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
    : BackendModel(triton_model)
{
}

TRITONSERVER_Error*
ModelState::ValidateAndParseModelConfig()
{
  RETURN_IF_ERROR(ValidateAndParseInputs());
  RETURN_IF_ERROR(ValidateAndParseOutputs());

  triton::common::TritonJson::Value params;
  bool is_unknown = true;
  if (model_config_.Find("parameters", &params)) {
    common::TritonJson::Value xrt;
    if (params.Find("xrt", &xrt)) {
      std::string xrt_str;
      RETURN_IF_ERROR(xrt.MemberAsString("string_value", &xrt_str));
      this->xrt_kind_ = ParseXrtKind(xrt_str, &is_unknown);
    }
  }
  if (is_unknown) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        "xrt tag is unknown, oneflow runtime will be used");
  }

  return nullptr;  // success
}

const std::vector<std::string>&
ModelState::InputNames() const
{
  return input_names_;
}

const std::vector<std::string>&
ModelState::OutputNames() const
{
  return output_names_;
}

const std::unordered_map<std::string, InputOutputAttribute>&
ModelState::InputAttributes() const
{
  return input_attribute_;
}

const std::unordered_map<std::string, InputOutputAttribute>&
ModelState::OutputAttributes() const
{
  return output_attribute_;
}

TRITONSERVER_Error*
ModelState::ValidateAndParseInputs()
{
  common::TritonJson::Value inputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  for (size_t io_index = 0; io_index < inputs.ArraySize(); ++io_index) {
    common::TritonJson::Value input;
    const char* input_name = nullptr;
    size_t input_name_len;
    std::string input_dtype_str;
    TRITONSERVER_DataType input_dtype;
    std::vector<int64_t> input_shape;
    triton::common::TritonJson::Value reshape;

    // parse
    RETURN_IF_ERROR(inputs.IndexAsObject(io_index, &input));
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));
    std::string input_name_str = std::string(input_name);
    RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype_str));
    if (input.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(backend::ParseShape(reshape, "shape", &input_shape));
    } else {
      RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
    }
    if (input_dtype_str.rfind("TYPE_", 0) != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, "DataType should start with TYPE_");
    }
    if (input_name_str.rfind("INPUT_", 0) != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "input name should start with INPUT_");
    }
    size_t input_index;
    try {
      input_index = std::atoi(input_name_str.substr(6).c_str());
      if (input_index >= inputs.ArraySize() || input_index < 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "input index should be in range [0, input_size)");
      }
    }
    catch (std::exception& ex) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "input name should follow naming convention: INPUT_<index>");
    }

    // store
    input_names_.push_back(input_name_str);
    input_dtype = TRITONSERVER_StringToDataType(
        input_dtype_str.substr(strlen("TYPE_")).c_str());
    input_attribute_[input_name_str] =
        InputOutputAttribute{input_dtype, input_shape, input_index};
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateAndParseOutputs()
{
  common::TritonJson::Value outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));
  for (size_t io_index = 0; io_index < outputs.ArraySize(); ++io_index) {
    common::TritonJson::Value output;
    const char* output_name = nullptr;
    size_t output_name_len;
    std::string output_dtype_str;
    TRITONSERVER_DataType output_dtype;
    std::vector<int64_t> output_shape;
    triton::common::TritonJson::Value reshape;

    // parse
    RETURN_IF_ERROR(outputs.IndexAsObject(io_index, &output));
    RETURN_IF_ERROR(
        output.MemberAsString("name", &output_name, &output_name_len));
    std::string output_name_str = std::string(output_name);
    RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype_str));
    if (output.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(backend::ParseShape(reshape, "shape", &output_shape));
    } else {
      RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));
    }
    if (output_dtype_str.rfind("TYPE_", 0) != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, "DataType should start with TYPE_");
    }
    if (output_name_str.rfind("OUTPUT_", 0) != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "output name should start with OUTPUT_");
    }
    size_t output_index;
    try {
      output_index = std::atoi(output_name_str.substr(7).c_str());
      if (output_index >= outputs.ArraySize() || output_index < 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "output index should be in the range [0, output_size)");
      }
    }
    catch (std::exception& ex) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "output name should follow naming convention: OUTPUT_<index>");
    }

    // store
    output_names_.push_back(output_name_str);
    output_dtype = TRITONSERVER_StringToDataType(
        output_dtype_str.substr(strlen("TYPE_")).c_str());
    output_attribute_[output_name_str] =
        InputOutputAttribute{output_dtype, output_shape, output_index};
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const oneflow_api::Device device,
    std::unique_ptr<oneflow_api::Graph>* graph)
{
  const std::string model_path =
      JoinPath({RepositoryPath(), std::to_string(Version()), "model"});

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + model_path +
            "' for model instance '" + Name() + "'");
  }

  graph->reset(
      new oneflow_api::Graph(oneflow_api::Graph::Load(model_path, device)));
  if (MaxBatchSize() > 0) {
    (*graph)->set_batch_size(MaxBatchSize());
  }
  if (IsXrtTensorrt(xrt_kind_)) {
    (*graph)->enable_tensorrt();
  }
  if (IsXrtOpenvino(xrt_kind_)) {
    (*graph)->enable_openvino();
  }

  return nullptr;
}

}}}  // namespace triton::backend::oneflow
