cmake_minimum_required(VERSION 3.5)

project(daintSkeleton)
set(PROJECT_NAME daintSkeleton)

#Destination
if (BINDIR)
  set(RUNTIME_DEST ${BINDIR})
else ()
  set(RUNTIME_DEST ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif ()

#cmake refs
set(PRJ_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PRJ_CMAKE_DIR})

#local ref
set(PRJ_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#################################
#  Generic Compilation options  #
#################################

#Compiler must support c++11
add_definitions(${CMAKE_CXX_FLAGS} "-std=c++14")

if (UNIX)
  add_definitions(${CMAKE_CXX_FLAGS_DEBUG} "-Wall -O0 -ggdb")
  add_definitions(${CMAKE_CXX_FLAGS_RELEASE} "-O3 -mtune=native -march=native")
endif ()

##############################
#  Cuda Compilation options  #
##############################

find_package(CUDA 8.0)

if (CUDA_FOUND AND USE_CUDA)
  add_definitions(-DUSE_CUDA)
  
  if (THRUST_BACKEND STREQUAL "OMP") 
    list(APPEND CUDA_NVCC_FLAGS
      -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
      -Xcompiler -fopenmp)
    list(APPEND cuda_deps gomp)
  else () #(THRUST_BACKEND STREQUAL "CUDA")
    list(APPEND CUDA_NVCC_FLAGS
      -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
  endif ()

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
  #runtime libs for cuda
  list(APPEND cuda_deps cudadevrt nvToolsExt)
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif ()

############
# MPI Part #
############
find_package(MPI)
if (MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
  #runtime libs for mpi
  set(mpi_deps ${MPI_CXX_LIBRARIES})
endif ()
#############
# /MPI Part #
#############

###############
# Boost part  #
###############
find_package(Boost COMPONENTS mpi serialization) 
if (Boost_FOUND)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mpi/include)#tmp
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
  #runtime libs for boost
 set(boost_deps mboost_mpi boost_serialization) #tmp
endif ()
################
# /Boost part  #
################

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
  ENABLE_IDS warning style performance portability information
  INCLUDES ${CURRENT_SOURCE_DIR}
  PROJECT_NAME ${PROJECT_NAME}
)

#####################################
#  Adding various code directories  #
#####################################

add_subdirectory(mpi)
add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)
add_subdirectory(scripts)

if (USE_REPORT)
  add_subdirectory(report)
endif ()
