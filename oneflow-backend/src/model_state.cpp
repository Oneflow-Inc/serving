#include "model_state.h"
#include <cstddef>

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

    const char* input_name = NULL;
    size_t input_name_len;
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));

    const char* output_name = NULL;
    size_t output_name_len;
    RETURN_IF_ERROR(
        output.MemberAsString("name", &output_name, &output_name_len));
    output_names_.emplace_back(output_name);

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

const std::vector<const char*>&
ModelState::GetOutputNames() const
{
  return output_names_;
}

}}}  // namespace triton::backend::oneflow
