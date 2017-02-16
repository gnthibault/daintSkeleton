cmake_minimum_required(VERSION 3.5)

project(test)

include_directories(../src/)

set(BIN_TEST_DIR ${RUNTIME_DEST}/test)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_TEST_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BIN_TEST_DIR})

enable_testing()

#Add various app tests
add_subdirectory(src)
