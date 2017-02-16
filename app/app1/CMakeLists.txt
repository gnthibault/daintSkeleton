cmake_minimum_required(VERSION 3.5)

##################
#  Generic Part  #
##################

set(OUTPUT_NAME  app1)

###############
#  Src files  #
###############

set(sources
  app1.cpp
)

set(headers
)

####################
#  Lib Dependancy  #
####################

include_directories( "../../src" )

################
#  Executable  #
################

add_executable(${OUTPUT_NAME} ${sources})

target_link_libraries(${OUTPUT_NAME} lib)

set_property(TARGET ${OUTPUT_NAME}
  PROPERTY OUTPUT_NAME ${OUTPUT_NAME})

install(TARGETS ${OUTPUT_NAME}
  RUNTIME DESTINATION ${RUNTIME_DEST})
