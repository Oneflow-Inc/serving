cmake_minimum_required(VERSION 3.18.0)

project(OneFlowServing)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

option(BUILD_CUDA "" OFF)
option(USE_TRITON "" ON)

option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_RELATED_REPO_TAG r21.10 CACHE STRING "Tag for triton-inference-server/(common|core|backend) repo")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(oneflow)

if(USE_TRITON)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    message(FATAL_ERROR "Clang is not supported when USE_TRITON=ON.")
  endif()
  include(triton)
else()
  message(FATAL_ERROR "Compiling without triton is not yet supported.")
endif()