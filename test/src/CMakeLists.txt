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


if (CUDA_FOUND AND USE_CUDA)
  add_cuda_lib_executable(cudalibtest1 ${lib_deps})
  add_cuda_lib_executable(cudalibtest2 ${lib_deps})
endif ()

##############
#  Unit test #
##############
add_test(NAME libtest1 COMMAND libtest1)
add_test(NAME libtest2 COMMAND libtest2)

if (CUDA_FOUND AND USE_CUDA)
  add_test(NAME cudalibtest1 COMMAND cudalibtest1)
  add_test(NAME cudalibtest2 COMMAND cudalibtest2)
endif ()

##################
#  Config files  #
##################
set(DATA_FILES
  "${TEST_ROOT_DIR}/common/data/data.csv"
)

install(FILES ${DATA_FILES}
  DESTINATION ${BIN_TEST_DIR} COMPONENT data)
