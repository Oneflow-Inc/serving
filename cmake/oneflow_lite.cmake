find_path(ONEFLOW_LITE_INCLUDE_DIR
  oneflow-lite/core/executable.h
  PATHS ${ONEFLOW_LITE_ROOT} ${ONEFLOW_LITE_ROOT}/include
        $ENV{ONEFLOW_LITE_ROOT} $ENV{ONEFLOW_LITE_ROOT}/include)

if (NOT ONEFLOW_LITE_INCLUDE_DIR)
  message(
    FATAL_ERROR
    "Unable to find path oneflow_lite/core/executable.h. You can set ONEFLOW_LITE_ROOT to specify the search path"
  )
endif()

function(FIND_LITE_LIBRARY lib_name _LIBRARIES)
  find_library(${lib_name}_LIBRARIES
    NAMES ${lib_name}
    PATHS ${ONEFLOW_LITE_ROOT} ${ONEFLOW_LITE_ROOT}/lib
          $ENV{ONEFLOW_LITE_ROOT} $ENV{ONEFLOW_LITE_ROOT}/lib)
  if (NOT ${lib_name}_LIBRARIES)
    message(
      FATAL_ERROR
      "Unable to find library ${lib_name}. You can set ONEFLOW_LITE_ROOT to specify the search path"
    )
  endif()
  set(${_LIBRARIES} ${${lib_name}_LIBRARIES} PARENT_SCOPE)
endfunction()

find_lite_library(oneflow-lite-runtime _LIBRARIES)
list(APPEND ONEFLOW_LITE_LIBRARIES ${_LIBRARIES})

message(STATUS "ONEFLOW_LITE_INCLUDE_DIR: ${ONEFLOW_LITE_INCLUDE_DIR}")
message(STATUS "ONEFLOW_LITE_LIBRARIES: ${ONEFLOW_LITE_LIBRARIES}")
