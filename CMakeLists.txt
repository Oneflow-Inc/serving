cmake_minimum_required(VERSION 3.18.0)

project(OneFlowServing)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

option(BUILD_CUDA "" OFF)
option(USE_TRITON "" ON)
option(USE_XLA "" OFF)
option(USE_TENSORRT "" OFF)
option(USE_OPENVINO "" OFF)

option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(THIRD_PARTY_MIRROR "" CACHE STRING "")
set(TRITON_RELATED_REPO_TAG r21.10 CACHE STRING "Tag for triton-inference-server/(common|core|backend) repo")
message(STATUS ${TRITON_RELATED_REPO_TAG})
if(NOT ${TRITON_RELATED_REPO_TAG} STREQUAL "r21.10")
  message(FATAL_ERROR "Only support triton with tag r21.10.")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(util)

find_package(oneflow)

if(USE_XLA OR USE_TENSORRT OR USE_OPENVINO)
  add_definitions(-DUSE_XRT=1)
  include(oneflow_xrt)
endif()

if(USE_TRITON)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    message(FATAL_ERROR "Clang is not supported when USE_TRITON=ON.")
  endif()
  include(triton)
else()
  message(FATAL_ERROR "Compiling without triton is not yet supported.")
endif()
