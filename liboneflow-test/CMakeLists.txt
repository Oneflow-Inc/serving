cmake_minimum_required(VERSION 3.18)
project(oneflow-test)

find_package(oneflow)
find_package(Threads )

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OneFlow::liboneflow Threads::Threads)
