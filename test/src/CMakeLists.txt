cmake_minimum_required(VERSION 3.5)

####################
#  Lib Dependancy  #
####################

include_directories( "../../src" )
set(lib_deps lib) 

################
#  Build test  #
################
add_lib_executable(libtest1 ${lib_deps})
add_lib_executable(libtest2 ${lib_deps})


if (CUDA_FOUND)
  add_cuda_lib_executable(cudalibtest1 ${lib_deps})
  add_cuda_lib_executable(cudalibtest2 ${lib_deps})
endif ()

##############
#  Unit test #
##############
add_test(NAME libtest1 COMMAND libtest1)
add_test(NAME libtest2 COMMAND libtest2)

if (CUDA_FOUND)
  add_test(NAME cudalibtest1 COMMAND cudalibtest1)
  add_test(NAME cudalibtest2 COMMAND cudalibtest2)
endif ()


##################
#  Config files  #
##################
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../../common/data/data.csv"
  "${BIN_TEST_DIR}/../../data/data.csv") COPYONLY)
