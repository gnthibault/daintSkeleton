cmake_minimum_required(VERSION 3.5)

project(daintSkeleton)

#Destination
if (BINDIR)
  set(RUNTIME_DEST ${BINDIR})
else ()
  set(RUNTIME_DEST ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif ()

#cmake refs
set(PRJ_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PRJ_CMAKE_DIR})


#################################
#  Generic Compilation options  #
#################################

#Compiler must support c++11
ADD_DEFINITIONS(${CMAKE_CXX_FLAGS} "-std=c++11")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  if (UNIX)
    ADD_DEFINITIONS (${CMAKE_CXX_FLAGS} "-Wall -O0 -ggdb")
  endif ()
else ()
  if (UNIX)
    ADD_DEFINITIONS (${CMAKE_CXX_FLAGS} "-O3 -mtune=native -march=native")
  endif ()
endif ()

##############################
#  Cuda Compilation options  #
##############################

find_package(CUDA 8.0)

if (CUDA_FOUND AND USE_CUDA)
  add_definitions(-DUSE_CUDA)
  
  #Eventually add some code instrumentation
  if (USE_NVTX)
    add_definitions(-DUSE_NVTX)
  endif ()

  #generic compilation flag, defines target architecture, type of build, ...
  list(APPEND CUDA_NVCC_FLAGS
    -gencode arch=compute_60,code=sm_60
    -std=c++11
    --expt-extended-lambda
    -rdc=true
    -Xcompiler -fPIC)

  set(CUDA_SEPARABLE_COMPILATION ON)

  if (CMAKE_BUILD_TYPE STREQUAL "Debug") 
    list(APPEND CUDA_NVCC_FLAGS 
      --device-debug
      --generate-line-info
      -g
      -G
      -DTHRUST_DEBUG )
  else ()
    list(APPEND CUDA_NVCC_FLAGS
      -use_fast_math
      -O3)
  endif ()
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif ()

#############
#  Testing  #
#############
enable_testing()

########################################
####        Doxygen Part            ####
########################################

# Add a target to generate API documentation with Doxygen
find_package(Doxygen)
if (DOXYGEN_FOUND)
  set(DOC_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/doc")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(
    doc
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOC_INSTALL_DIR}
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif ()

##############
#  Cppcheck  #
##############

# Add a target to generate Cppcheck report
include(${PRJ_CMAKE_DIR}/gen_cppcheck.cmake)
GENERATE_CPPCHECK(SOURCEDIRS src app test 
  ENABLE_IDS warning style performance portability information)

#####################################
#  Adding various code directories  #
#####################################

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)
add_subdirectory(scripts)

