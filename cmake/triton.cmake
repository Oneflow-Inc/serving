include(FetchContent)

set(REPO_COMMON_URL https://github.com/triton-inference-server/common/archive/refs/heads/${TRITON_RELATED_REPO_TAG}.zip)
set(REPO_CORE_URL https://github.com/triton-inference-server/core/archive/refs/heads/${TRITON_RELATED_REPO_TAG}.zip)
set(REPO_BACKEND_URL https://github.com/triton-inference-server/backend/archive/refs/heads/${TRITON_RELATED_REPO_TAG}.zip)

if(DEFINED THIRD_PARTY_MIRROR)
  use_mirror(VARIABLE REPO_COMMON_URL URL ${REPO_COMMON_URL})
  use_mirror(VARIABLE REPO_CORE_URL URL ${REPO_CORE_URL})
  use_mirror(VARIABLE REPO_BACKEND_URL URL ${REPO_BACKEND_URL})
endif()

if(${TRITON_RELATED_REPO_TAG} STREQUAL "r21.10")
    set(REPO_COMMON_MD5 72bf32b638fe6a9e9877630cb099fc1a)
    set(REPO_CORE_MD5 59d97b3e5d40ea58c9f685b6ecb0771a)
    set(REPO_BACKEND_MD5 2ae374cf913fc5b348b6552858fb7e7b)
else()
  message(FATAL_ERROR "Only support triton with tag r21.10.")
endif()


FetchContent_Declare(
  repo-common
  URL ${REPO_COMMON_URL}
  URL_HASH MD5=${REPO_COMMON_MD5}
)

FetchContent_Declare(
  repo-core
  URL ${REPO_CORE_URL}
  URL_HASH MD5=${REPO_CORE_MD5}
)

FetchContent_Declare(
  repo-backend
  URL ${REPO_BACKEND_URL}
  URL_HASH MD5=${REPO_BACKEND_MD5}
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
    ${ONEFLOW_XRT_INCLUDE_DIR}
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
    -Wl,--no-as-needed
    ${ONEFLOW_XRT_LIBRARIES}
    -Wl,--as-needed
)

set_target_properties(
  triton-oneflow-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_oneflow
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_oneflow.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_oneflow.ldscript"
)
