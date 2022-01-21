include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_RELATED_REPO_TAG}
  GIT_SHALLOW ON
)

FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_RELATED_REPO_TAG}
  GIT_SHALLOW ON
)

FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_RELATED_REPO_TAG}
  GIT_SHALLOW ON
)

FetchContent_MakeAvailable(repo-common repo-core repo-backend)

set(triton_oneflow_backend_include_dir ${PROJECT_SOURCE_DIR}/include/triton)
set(triton_oneflow_backend_source_dir ${PROJECT_SOURCE_DIR}/src/triton)

configure_file(${triton_oneflow_backend_source_dir}/libtriton_oneflow.ldscript libtriton_oneflow.ldscript COPYONLY)

file(GLOB triton_oneflow_backend_sources ${triton_oneflow_backend_source_dir}/*.cpp)

add_library(
  triton-oneflow-backend SHARED
  ${triton_oneflow_backend_sources}
)

add_library(
  TritonOneFlowBackend::triton-oneflow-backend ALIAS triton-oneflow-backend
)

target_include_directories(
  triton-oneflow-backend
  PRIVATE
    ${triton_oneflow_backend_include_dir}
)

target_compile_features(triton-oneflow-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-oneflow-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

target_link_libraries(
  triton-oneflow-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    OneFlow::liboneflow
)

set_target_properties(
  triton-oneflow-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_oneflow
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_oneflow.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_oneflow.ldscript"
)