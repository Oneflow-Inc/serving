#include "model_state.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "oneflow_utils.h"
#ifdef USE_XRT
#include "oneflow_xrt_utils.h"
#endif  // USE_XRT
#include "triton/backend/backend_common.h"
#include "triton/common/triton_json.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace oneflow {

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
    bool auto_complete_config = false;
    RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
        triton_model, &auto_complete_config));
    if (auto_complete_config) {
      RETURN_IF_ERROR((*state)->AutoCompleteConfig());

      // Update model config
      triton::common::TritonJson::WriteBuffer json_buffer;
      (*state)->ModelConfig().Write(&json_buffer);

      TRITONSERVER_Message* message;
      RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
          &message, json_buffer.Base(), json_buffer.Size()));
      RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(triton_model, 1, message));
    }
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
ModelState::AutoCompleteConfig()
{
  std::unique_ptr<oneflow_api::Graph> graph;
  LoadModel(oneflow_api::Device("cpu"), &graph);
  auto input_infos = graph->GetInputInfos();
  auto output_infos = graph->GetOutputInfos();

  AutoCompleteInputsAndOutputs(true, input_infos);
  AutoCompleteInputsAndOutputs(false, output_infos);
  AutoCompleteMaxBatchSize();

  triton::common::TritonJson::WriteBuffer buffer;
  ModelConfig().PrettyWrite(&buffer);
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      std::string("Auto-completed config: " + buffer.Contents()).c_str());
  return nullptr;
}

TRITONSERVER_Error*
ModelState::AutoCompleteInputsAndOutputs(
    bool is_input, oneflow_api::InputOutputInfos& input_output_infos)
{
  triton::common::TritonJson::Value existing_ios;
  const char* key = is_input ? "input" : "output";
  bool found_ios = ModelConfig().Find(key, &existing_ios);

  std::unordered_set<std::string> existing_io_names;
  for (size_t i = 0; i < existing_ios.ArraySize(); ++i) {
    triton::common::TritonJson::Value value;
    existing_ios.IndexAsObject(i, &value);
    std::string name;
    value.MemberAsString("name", &name);
    existing_io_names.insert(name);
  }

  if (!found_ios) {
    existing_ios = triton::common::TritonJson::Value(
        ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  }

  int index = 0;
  for (const auto& info : input_output_infos) {
    std::string input_output_name;
    if (is_input) {
      input_output_name = std::string("INPUT_") + std::to_string(index);
    } else {
      input_output_name = std::string("OUTPUT_") + std::to_string(index);
    }
    if (existing_io_names.find(input_output_name) != existing_io_names.end()) {
      index += 1;
      continue;
    }

    TRITONSERVER_DataType data_type =
        ConvertOneFlowTypeToTritonType(info.second.datatype_);
    const char* data_type_str = TRITONSERVER_DataTypeString(data_type);
    std::vector<int64_t> dims_vector =
        OfShapeToVector(info.second.input_output_shape_);

    triton::common::TritonJson::Value io(
        ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_ERROR(io.AddString("name", input_output_name));

    RETURN_IF_ERROR(io.AddString(
        "data_type", ("TYPE_" + std::string(data_type_str)).c_str()));
    triton::common::TritonJson::Value dims(
        ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
    for (const int64_t& dim : dims_vector) {
      RETURN_IF_ERROR(dims.AppendInt(dim));
    }
    RETURN_IF_ERROR(io.Add("dims", std::move(dims)));
    RETURN_IF_ERROR(existing_ios.Append(std::move(io)));

    index += 1;
  }

  if (!found_ios) {
    ModelConfig().Add(key, std::move(existing_ios));
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::AutoCompleteMaxBatchSize()
{
  triton::common::TritonJson::Value mbs_value;
  if (!ModelConfig().Find("max_batch_size", &mbs_value)) {
    mbs_value.SetInt(1);
    SetMaxBatchSize(1);
  }
  return nullptr;
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
#ifdef USE_XRT
  if (!IsXrtOneFlow(xrt_kind_)) {
    (*graph)->RegisterJobPass([this](const std::string& job) -> std::string {
      return oneflow_xrt::Transform(job, this->xrt_kind_);
    });
  }
#else
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "unable to use XRT since it was not compiled with TENSORRT, OPENVINO or "
      "XLA, so use oneflow instead");
#endif
  return nullptr;
}

}}}  // namespace triton::backend::oneflow
