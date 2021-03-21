cmake_minimum_required(VERSION 2.8)

project(cpplex)

set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall -Wextra")

include_directories(include/cpplex)

add_executable(solver src/solver.cc)
add_executable(conicalhull src/conicalhull.cc)

find_package(Eigen3 3.3 NO_MODULE)
if (TARGET Eigen3::Eigen)
  add_executable(solver-eigen src/solver-eigen.cc)
  target_link_libraries(solver-eigen Eigen3::Eigen)
endif (TARGET Eigen3::Eigen)

