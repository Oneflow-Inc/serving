include(FetchContent)

set(REPO_COMMON_URL https://github.com/triton-inference-server/common/archive/refs/heads/${TRITON_RELATED_REPO_TAG}.zip)
set(REPO_CORE_URL https://github.com/triton-inference-server/core/archive/refs/heads/${TRITON_RELATED_REPO_TAG}.zip)
set(REPO_BACKEND_URL https://github.com/triton-inference-server/backend/archive/refs/heads/${TRITON_RELATED_REPO_TAG}.zip)

if(DEFINED THIRD_PARTY_MIRROR)
  use_mirror(VARIABLE REPO_COMMON_URL URL ${REPO_COMMON_URL})
  use_mirror(VARIABLE REPO_CORE_URL URL ${REPO_CORE_URL})
  use_mirror(VARIABLE REPO_BACKEND_URL URL ${REPO_BACKEND_URL})
endif()

if(${TRITON_RELATED_REPO_TAG} STREQUAL "r23.10")
    set(REPO_COMMON_MD5 8183efa82f41c4964c26e9b839ef2760)
    set(REPO_CORE_MD5 ba92d1b9aa5154edb26fc9664224f9ae)
    set(REPO_BACKEND_MD5 c7a6a21353e8f00e61bd97afd8708c0a)
else()
  message(FATAL_ERROR "Only support triton with tag r23.10.")
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

if(BUILD_ONEFLOW_BACKEND)
  set(triton_oneflow_backend_include_dir ${PROJECT_SOURCE_DIR}/include/triton/backend_oneflow)
  set(triton_oneflow_backend_source_dir ${PROJECT_SOURCE_DIR}/src/triton/backend_oneflow)
  
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
endif()

if(BUILD_ONEFLOW_LITE_BACKEND)  
  file(GLOB triton_oneflow_lite_backend_sources ${PROJECT_SOURCE_DIR}/src/triton/backend_oneflow_lite/*.cpp)
  add_library(
    triton-oneflow-lite-backend SHARED
    ${triton_oneflow_lite_backend_sources}
  )
  
  add_library(
    TritonOneFlowLiteBackend::triton-oneflow-lite-backend ALIAS triton-oneflow-lite-backend
  )
  
  target_include_directories(
    triton-oneflow-lite-backend
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include/triton/backend_oneflow_lite
    ${ONEFLOW_LITE_INCLUDE_DIR}
  )
  
  target_compile_features(triton-oneflow-lite-backend PRIVATE cxx_std_11)
  target_compile_options(
    triton-oneflow-lite-backend PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  )
  
  target_link_libraries(
    triton-oneflow-lite-backend
    PRIVATE
      triton-core-serverapi   # from repo-core
      triton-core-backendapi  # from repo-core
      triton-core-serverstub  # from repo-core
      triton-backend-utils    # from repo-backend
      -Wl,--no-as-needed
      ${ONEFLOW_LITE_LIBRARIES}
      -Wl,--as-needed
  )
  
  configure_file(${PROJECT_SOURCE_DIR}/src/triton/backend_oneflow_lite/libtriton_oneflow_lite.ldscript
                 libtriton_oneflow_lite.ldscript COPYONLY)
  set_target_properties(
    triton-oneflow-lite-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_oneflow_lite
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_oneflow_lite.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_oneflow_lite.ldscript"
  ) 
endif()
