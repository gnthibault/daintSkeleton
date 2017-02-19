cmake_minimum_required(VERSION 3.5)

#################
#  Generic Part #
#################

project(lib)
set(OUTPUT_NAME lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
###############
#  Cuda Part  #
###############
if (CUDA_FOUND AND USE_CUDA)
  set(sources_cuda
    Cuda/lib.cu
  )
  set(headers_cuda
    Cuda/lib.cu.h
  )
  #runtime libs for cuda
  set(cuda_deps cudadevrt nvToolsExt)
endif ()
################
#  /Cuda Part  #
################i

############
# MPI Part #
############
find_package(MPI)
if(MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
  #Allowing MPI specific code
  add_definitions(-DUSE_MPI)
  set(sources_mpi
  )
  set(headers_mpi
  )
  #runtime libs for mpi
  set(mpi_deps)
endif()
#############
# /MPI Part #
#############

###############
# Boost part  #
###############
find_package(Boost COMPONENTS mpi serialization) 

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS}) 
  set(boost_deps boost_mpi) #${Boost_LIBRARIES}
endif()
################
# /Boost part  #
################

###############
#  Src files  #
###############
set(sources
  lib.cpp
  # cuda source files
  ${sources_cuda}
  # mpi source  files
  ${sources_mpi}
)

set(headers
  lib.h
  # cuda header files
  ${headers_cuda}
  #mpi header files
  ${headers_mpi} 
)

################
#  Executable  #
################

if (CUDA_FOUND)
  cuda_add_library(${OUTPUT_NAME} SHARED ${sources} ${headers})
  #CUDA_ADD_CUBLAS_TO_TARGET(${OUTPUT_NAME})
else ()
  add_library(${OUTPUT_NAME} SHARED ${sources} ${headers})
endif ()

target_link_libraries(${OUTPUT_NAME} ${cuda_deps} ${mpi_deps})

set_property(TARGET ${OUTPUT_NAME}
  PROPERTY OUTPUT_NAME ${output_name})

install(TARGETS ${OUTPUT_NAME}
  LIBRARY DESTINATION ${RUNTIME_DEST}
)
