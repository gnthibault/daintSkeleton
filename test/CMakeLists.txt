cmake_minimum_required(VERSION 3.5)

project(test)

set(BIN_TEST_DIR ${RUNTIME_DEST}/test)
set(TEST_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#Add handmade macro to cmake path
list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(libtest)

#Add various app tests
add_subdirectory(src)
