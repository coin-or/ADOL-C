find_package(Doxygen)

if(DOXYGEN_FOUND)

  # Absolute path to manual in the source tree:
  set(PDF_SOURCE_PATH ${CMAKE_SOURCE_DIR}/ADOL-C/doc/adolc-manual.pdf)
  set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.in)
  set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/doxyfile)
  configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

  add_custom_target(doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating class documentation using Doxygen"
        VERBATIM)

endif()
