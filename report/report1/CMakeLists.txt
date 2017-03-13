cmake_minimum_required(VERSION 3.5)

project(report1)
set(PROJECT_NAME report1)

#file(GLOB_RECURSE contents_files RELATIVE ${CMAKE_SOURCE_DIR} contents/*.tex)

add_latex_document(
  report1.tex
  #INPUTS ${contents_files}
  #IMAGE_DIRS images
  BIBFILES biblio.bib
)
