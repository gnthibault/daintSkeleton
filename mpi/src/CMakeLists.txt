cmake_minimum_required(VERSION 3.5)

set(OUTPUT_NAME mboost_mpi)
###############
#  Src files  #
###############
set(sources
  broadcast.cpp
  computation_tree.cpp
  graph_communicator.cpp
  mpi_datatype_oarchive.cpp
  packed_skeleton_iarchive.cpp
  request.cpp
  cartesian_communicator.cpp
  content_oarchive.cpp
  group.cpp
  offsets.cpp
  packed_skeleton_oarchive.cpp
  text_skeleton_oarchive.cpp
  environment.cpp
  intercommunicator.cpp
  packed_iarchive.cpp
  point_to_point.cpp
  timer.cpp
  communicator.cpp
  exception.cpp
  mpi_datatype_cache.cpp
  packed_oarchive.cpp
)

################
#  Executable  #
################

add_library(${OUTPUT_NAME} SHARED ${sources})

target_link_libraries(${OUTPUT_NAME}
  ${mpi_deps}
)

set_property(TARGET ${OUTPUT_NAME}
  PROPERTY OUTPUT_NAME ${output_name})

install(TARGETS ${OUTPUT_NAME}
  LIBRARY DESTINATION ${RUNTIME_DEST}
)
