find_path(ONEFLOW_XRT_INCLUDE_DIR
  oneflow_xrt/api/api_serving.h
  PATHS ${ONEFLOW_XRT_ROOT} ${ONEFLOW_XRT_ROOT}/include
        $ENV{ONEFLOW_XRT_ROOT} $ENV{ONEFLOW_XRT_ROOT}/include)

if (NOT ONEFLOW_XRT_INCLUDE_DIR)
  message(
    FATAL_ERROR
    "Unable to find path api/api_serving.h. You can set ONEFLOW_XRT_ROOT to specify the search path"
  )
endif()

function(FIND_XRT_LIBRARY lib_name _LIBRARIES)
  find_library(${lib_name}_LIBRARIES
    NAMES ${lib_name}
    PATHS ${ONEFLOW_XRT_ROOT} ${ONEFLOW_XRT_ROOT}/lib
          $ENV{ONEFLOW_XRT_ROOT} $ENV{ONEFLOW_XRT_ROOT}/lib)
  if (NOT ${lib_name}_LIBRARIES)
    message(
      FATAL_ERROR
      "Unable to find library ${lib_name}. You can set ONEFLOW_XRT_ROOT to specify the search path"
    )
  endif()
  set(${_LIBRARIES} ${${lib_name}_LIBRARIES} PARENT_SCOPE)
endfunction()

find_xrt_library(oneflow_xrt _LIBRARIES)
list(APPEND ONEFLOW_XRT_LIBRARIES ${_LIBRARIES})

if(USE_XLA)
  find_xrt_library(oneflow_xrt_xla _LIBRARIES)
  list(APPEND ONEFLOW_XRT_LIBRARIES ${_LIBRARIES})
endif()

if(USE_TENSORRT)
  find_xrt_library(oneflow_xrt_tensorrt _LIBRARIES)
  list(APPEND ONEFLOW_XRT_LIBRARIES ${_LIBRARIES})
endif()

if(USE_OPENVINO)
  find_xrt_library(oneflow_xrt_openvino _LIBRARIES)
  list(APPEND ONEFLOW_XRT_LIBRARIES ${_LIBRARIES})
endif()

message(STATUS "ONEFLOW_XRT_INCLUDE_DIR: ${ONEFLOW_XRT_INCLUDE_DIR}")
message(STATUS "ONEFLOW_XRT_LIBRARIES: ${ONEFLOW_XRT_LIBRARIES}")
