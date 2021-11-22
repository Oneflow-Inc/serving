#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <oneflow/api.h>
#include <oneflow/device.h>
#include <oneflow/dtype.h>
#include <oneflow/env.h>
#include <oneflow/nn.h>
#include <oneflow/shape.h>
#include <oneflow/tensor.h>
#include <string>
#include <thread>
#include <vector>
#include <cmath>

class EnvScope {
 public:
  EnvScope() { oneflow_api::initialize(); }
  ~EnvScope() { oneflow_api::release(); }
};

std::vector<int64_t> random_shape() {
  int shape_dims = rand() % 5 + 1;
  std::vector<int64_t> shape(shape_dims);
  for (int i = 0; i < shape_dims; i++) { shape[i] = rand() % 20 + 1; }
  return shape;
}

std::vector<float> random_data(int64_t count) {
  std::vector<float> data(count);
  for (int i = 0; i < count; i++) {
    data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5;
  }
  return data;
}

std::vector<float> relu_answer(std::vector<float> data, int64_t count) {
  std::vector<float> relu(count);
  for (int i = 0; i < count; i++) { relu[i] = data[i] > 0 ? data[i] : 0; }
  return relu;
}

int main(int argc, char** argv) {
  EnvScope scope;

  constexpr int thread_nums = 10;
  constexpr int forward_times = 1000;
  std::thread threads[thread_nums];
  for (int i = 0; i < thread_nums; i++) {
    threads[i] = std::thread([]() {
      std::vector<int64_t> shape_vec = random_shape();
      int64_t element_count =
          std::accumulate(shape_vec.begin(), shape_vec.end(), 1, std::multiplies<int64_t>());
      std::vector<float> data = random_data(element_count);
      std::vector<float> answer = relu_answer(data, element_count);

      oneflow_api::Shape shape(shape_vec);
      oneflow_api::Device device("cpu");
      oneflow_api::DType dtype = oneflow_api::DType::kFloat;
      oneflow_api::Tensor tensor =
          oneflow_api::Tensor::from_blob(data.data(), shape, device, dtype);

      for (int j = 0; j < forward_times; j++) {
        oneflow_api::Tensor result_tensor = oneflow_api::nn::relu(tensor);
        std::vector<float> result_data(element_count);
        oneflow_api::Tensor::to_blob(result_tensor, result_data.data());

        // shape
        auto result_shape = tensor.shape();
        if (result_shape != shape) {
          std::cout << "wrong shape" << std::endl;
          exit(-1);
        }

        // element count
        uint64_t result_element_count = result_shape.elem_cnt();
        if (result_element_count != element_count) {
          std::cout << result_element_count << " " << element_count << std::endl;
          std::cout << "wrong element count" << std::endl;
          exit(-1);
        }

        // check float array
        for (int64_t ele_id = 0; ele_id < element_count; ele_id++) {
          if (fabs(result_data[ele_id] - answer[ele_id]) > 0.001) {
            std::cout << data[ele_id] << " " << result_data[ele_id] << " " << answer[ele_id]
                      << std::endl;
            std::cout << "wrong result data" << std::endl;
            exit(-1);
          }
        }
      }
    });
  }

  for (int i = 0; i < thread_nums; i++) { threads[i].join(); }
  std::cout << "Test Passed" << std::endl;

  return 0;
}
