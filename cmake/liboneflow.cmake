message(STATUS 000)
include(FetchContent)

FetchContent_Declare(
  oneflow
  GIT_REPOSITORY https://github.com/Oneflow-Inc/oneflow.git
  GIT_TAG master
  GIT_SHALLOW ON
  BUILD_CUDA NO
  BUILD_TESTING NO
  BUILD_PYTHON ON
  BUILD_CPP_API ON
  WITH_MLIR ON
  THIRD_PARTY_MIRROR aliyun
  PIP_INDEX_MIRROR "https://pypi.tuna.tsinghua.edu.cn/simple"
)

FetchContent_MakeAvailable(oneflow)