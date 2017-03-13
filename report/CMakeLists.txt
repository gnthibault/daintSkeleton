cmake_minimum_required(VERSION 3.5)

set(REPORT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LATEX_OUTPUT_PATH "${PRJ_ROOT_DIR}/doc")

#cmake refs
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
INCLUDE(UseLATEX)

add_subdirectory(report1)
