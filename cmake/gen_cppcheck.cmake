# - Generate a cppcheck documentation for a project.
# The function GENERATE_CPPCHECK is provided to create a "cppcheck" target that
# performs static code analysis using the cppcheck utility program.
#
# GENERATE_CPPCHECK(SOURCES <sources to check...>
#                   SOURCEDIRS <source directories>
#                   [SUPPRESSION_FILE <file>]
#                   [ENABLE_IDS <id...>]
#                   [TARGET_NAME <name>]
#                   [INCLUDES <dir...>])
#
# Generates a target "cppcheck" that executes cppcheck on the specified sources.
# Sources may either be file names or directories containing files where all
# C++ files will be parsed automatically. Use directories whenever possible
# because there is a limitation in arguments to pass to the cppcheck binary.
# SUPPRESSION_FILE may be give additionally to specify suppressions for#
# cppcheck. The sources mentioned in the suppression file must be in the same
# format like given for SOURCES. This means if you specified them relative to
# CMAKE_CURRENT_SOURCE_DIR, then the same relative paths must be used in the
# suppression file.
# When SOURCEDIRS is given, cppcheck-htmlreport is run on the produced xml output. 
# Python 2.7 with pygments is required for this feature.
# ENABLE_IDS allows to specify which additional cppcheck check ids to execute,
# e.g. all or style. They are combined with AND.
# With TARGET_NAME a different name for the generated check target can be
# specified. This is useful if several calles to this function are made in one
# CMake project, as otherwise the target names collide.
# Additional include directories for the cppcheck program can be given with
# INCLUDES.
#
# cppcheck will be executed with CMAKE_CURRENT_SOURCE_DIR as working directory.
#
# This function can always be called, even if no cppcheck was found. Then no
# target is created.
#
# Copyright (C) 2011 by Johannes Wienke <jwienke at techfak dot uni-bielefeld dot de>
#
# This program is free software; you can redistribute it
# and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation;
# either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# Modified 28-08-2014 E.J Boks (Kiwanda Embedded Systemen) to run on CMake 3.0.1 in conjunction with cppcheck-htmlreport

GET_FILENAME_COMPONENT(GENERATE_CPPCHECK_MODULE_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)

FIND_PACKAGE(cppcheck)

FUNCTION(GENERATE_CPPCHECK)
  set(options )
  set(oneValueArgs SUPPRESSION_FILE TARGET_NAME)
  set(multiValueArgs ENABLE_IDS SOURCES SOURCEDIRS INCLUDES)

    IF(CPPCHECK_FOUND)
        CMAKE_PARSE_ARGUMENTS(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
        SET(TARGET_NAME "cppcheck")
        SET(TARGET_NAME_SUFFIX "")
        # parse target name
        LIST(LENGTH ARG_TARGET_NAME TARGET_NAME_LENGTH)
        IF(${TARGET_NAME_LENGTH} EQUAL 1)
            SET(TARGET_NAME ${ARG_TARGET_NAME})
            SET(TARGET_NAME_SUFFIX "-${ARG_TARGET_NAME}")
        ENDIF()
        
        SET(CPPCHECK_CHECKFILE "${CMAKE_CURRENT_BINARY_DIR}/cppcheck-files${TARGET_NAME_SUFFIX}")
        SET(CPPCHECK_REPORT_FILE "${CMAKE_CURRENT_BINARY_DIR}/cppcheck-report${TARGET_NAME_SUFFIX}.xml")
        SET(CPPCHECK_WRAPPER_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/cppcheck${TARGET_NAME_SUFFIX}.cmake")
     
        # write a list file containing all sources to check for the call to
        # cppcheck
        message("SOURCES is currently ${ARG_SOURCES}")
        message("SOURCEDIRS is currently ${ARG_SOURCEDIRS}")

        IF(ARG_SOURCES)
          SET(SOURCELIST ${ARG_SOURCES})
        ELSE(ARG_SOURCES)
          SET(SOURCELIST ${ARG_SOURCEDIRS})
        ENDIF(ARG_SOURCES)
        SET(SOURCE_ARGS "")
        FOREACH(SOURCE ${SOURCELIST})
	    #Get absolute path of file, for CI tool report publication for instance
	    get_filename_component(CURRENT_SOURCE_ABS_PATH ${SOURCE} ABSOLUTE)
	    LIST(APPEND SOURCE_ARGS ${CURRENT_SOURCE_ABS_PATH})
            message("SOURCE is currently ${SOURCE_ARGS}")
        ENDFOREACH()
        
        # prepare a cmake wrapper to write the stderr output of cppcheck to
        # the result file
        
        # suppression argument
        LIST(LENGTH ARG_SUPPRESSION_FILE SUPPRESSION_FILE_LENGTH)
        IF(${SUPPRESSION_FILE_LENGTH} EQUAL 1)
            GET_FILENAME_COMPONENT(ABS "${ARG_SUPPRESSION_FILE}" ABSOLUTE)
            MESSAGE(STATUS "Using suppression file ${ABS}")
            SET(SUPPRESSION_ARGUMENT --suppressions)
            SET(SUPPRESSION_FILE "\"${ABS}\"")
        ENDIF()
        
        # includes
        SET(INCLUDE_ARGUMENTS "")
        FOREACH(INCLUDE ${ARG_INCLUDES})
            SET(INCLUDE_ARGUMENTS "${INCLUDE_ARGUMENTS} \"-I${INCLUDE}\"")
        ENDFOREACH()
        
        # enabled ids
        SET(ID_LIST "")
        FOREACH(ID ${ARG_ENABLE_IDS})
            SET(ID_LIST "${ID_LIST},${ID}")
        ENDFOREACH()
        IF(ID_LIST)
            STRING(LENGTH ${ID_LIST} LIST_LENGTH)
            MATH(EXPR FINAL_LIST_LENGTH "${LIST_LENGTH} - 1")
            STRING(SUBSTRING ${ID_LIST} 1 ${FINAL_LIST_LENGTH} FINAL_ID_LIST)
            SET(IDS_ARGUMENT "\"--enable=${FINAL_ID_LIST}\"")
        ELSE()
            SET(IDS_ARGUMENT "")
        ENDIF()
     
        if(ARG_SOURCEDIRS)
            set(CPPCHECK_REPORT_DIR "${CMAKE_CURRENT_BINARY_DIR}/cppcheckdir-report${TARGET_NAME_SUFFIX}")   
            file(WRITE ${CPPCHECK_WRAPPER_SCRIPT}
"
EXECUTE_PROCESS(COMMAND \"${CPPCHECK_EXECUTABLE}\" ${INCLUDE_ARGUMENTS} ${SUPPRESSION_ARGUMENT} ${SUPPRESSION_FILE} ${IDS_ARGUMENT} --inline-suppr --xml ${SOURCE_ARGS}
                RESULT_VARIABLE CPPCHECK_EXIT_CODE
                ERROR_VARIABLE ERROR_OUT
                WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\")
IF(NOT CPPCHECK_EXIT_CODE EQUAL 0)
    MESSAGE(FATAL_ERROR \"Error executing cppcheck for target ${TARGET}, return code: \${CPPCHECK_EXIT_CODE}\")
else(NOT CPPCHECK_EXIT_CODE EQUAL 0)
  EXECUTE_PROCESS(COMMAND \"${CPPCHECK_EXECUTABLE}-htmlreport\" \"--file=${CPPCHECK_REPORT_FILE}\"  \"--report-dir=${CPPCHECK_REPORT_DIR}\" \"--source-dir=${ARG_SOURCEDIRS}\")
ENDIF()
IF(ERROR_OUT)
    MESSAGE(\"Detected errors:\\n\${ERROR_OUT}\")
ENDIF()
FILE(WRITE \"${CPPCHECK_REPORT_FILE}\" \"\${ERROR_OUT}\")
"
            )
        else(ARG_SOURCEDIRS)
        file(WRITE ${CPPCHECK_WRAPPER_SCRIPT}
"
EXECUTE_PROCESS(COMMAND \"${CPPCHECK_EXECUTABLE}\" ${INCLUDE_ARGUMENTS} ${SUPPRESSION_ARGUMENT} ${SUPPRESSION_FILE} ${IDS_ARGUMENT} --inline-suppr --xml-version=2 --xml ${SOURCE_ARGS}
                RESULT_VARIABLE CPPCHECK_EXIT_CODE
                ERROR_VARIABLE ERROR_OUT
                WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\")
IF(NOT CPPCHECK_EXIT_CODE EQUAL 0)
    MESSAGE(FATAL_ERROR \"Error executing cppcheck for target ${TARGET}, return code: \${CPPCHECK_EXIT_CODE}\")
ENDIF()
IF(ERROR_OUT)
    MESSAGE(\"Detected errors:\\n\${ERROR_OUT}\")
ENDIF()
FILE(WRITE \"${CPPCHECK_REPORT_FILE}\" \"\${ERROR_OUT}\")
"
            )
        endif(ARG_SOURCEDIRS)
     
        ADD_CUSTOM_TARGET(${TARGET_NAME} ${CMAKE_COMMAND} -P "${CPPCHECK_WRAPPER_SCRIPT}"
                          COMMENT "Generating cppcheck result ${TARGET_NAME}")
                          
        MESSAGE(STATUS "Generating cppcheck target with name ${TARGET_NAME}")

    ENDIF()

ENDFUNCTION()

