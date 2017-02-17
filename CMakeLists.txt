cmake_minimum_required(VERSION 3.5)

project(daintSkeleton)

#Destination
if (BINDIR)
  set(RUNTIME_DEST ${BINDIR})
else ()
  set(RUNTIME_DEST ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif ()
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

if (CUDA_FOUND)
  add_definitions(-DGPU_CUDA)
  
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

#####################################
#  Adding various code directories  #
#####################################

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)
add_subdirectory(scripts)

