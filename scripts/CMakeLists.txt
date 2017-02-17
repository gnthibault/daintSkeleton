cmake_minimum_required(VERSION 3.5)

#################
#  Generic Part #
#################

project(launchScript)
set(PROJECT_NAME launchScript)

##################
#  Config files  #
##################
set(CONFIG_FILES
  launchApp1.sh
  launchApp1WithProfiling.sh
)

install(FILES ${CONFIG_FILES}
  DESTINATION ${RUNTIME_DEST} COMPONENT config)
