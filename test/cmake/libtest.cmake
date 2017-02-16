
macro(add_cuda_lib_executable exename)
  set(dependencyName ${ARGN})
  cuda_add_executable(${exename} "${exename}.cu")
  target_link_libraries(${exename} ${dependencyName} )
endmacro()

macroo(add_lib_executable exename)
  set(dependencyName ${ARGN})
  add_executable(${exename} "${exename}.cpp")
  target_link_libraries(${exename} ${dependencyName} )
endmacro()

