find_package(Doxygen COMPONENTS doxygen dot)

if(NOT Doxygen_FOUND)
  message(STATUS "Doxygen was not found; documentation targets are unavailable")
  return()
endif()

set(ADOLC_DOC_EXAMPLE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples")
set(ADOLC_EXAMPLE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../examples")
set(ADOLC_ADDITIONAL_EXAMPLE_SOURCE_DIR
    "${ADOLC_EXAMPLE_SOURCE_DIR}/additional_examples")
set(ADOLC_DOC_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ADOLC_DOC_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")
set(ADOLC_MANUAL_PDF "${CMAKE_CURRENT_SOURCE_DIR}/../adolc-manual.pdf")

find_package(Threads REQUIRED)

function(add_tested_documentation_example example source)
  set(target "adolc_doc_${example}")
  add_executable(${target} EXCLUDE_FROM_ALL "${source}")
  target_link_libraries(${target} PRIVATE adolc Threads::Threads)

  set(output "${ADOLC_DOC_GENERATED_DIR}/${example}.out")
  set(runtime_directory
      "${CMAKE_CURRENT_BINARY_DIR}/example-runtime/${example}")
  add_custom_command(
    OUTPUT "${output}"
    COMMAND
      ${CMAKE_COMMAND}
      "-DEXAMPLE_EXECUTABLE=$<TARGET_FILE:${target}>"
      "-DOUTPUT_FILE=${output}"
      "-DRUNTIME_DIRECTORY=${runtime_directory}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/RunExample.cmake"
    DEPENDS ${target} "${CMAKE_CURRENT_SOURCE_DIR}/RunExample.cmake"
    COMMENT "Running and validating documentation example ${example}"
    VERBATIM)
  list(APPEND documentation_example_outputs "${output}")
  set(documentation_example_outputs
      "${documentation_example_outputs}"
      PARENT_SCOPE)
endfunction()

set(documentation_examples
  taped_gradient
  taped_jacobian
  higher_order
  tapeless_forward
  parallel_evaluation)

foreach(example IN LISTS documentation_examples)
  add_tested_documentation_example(
    "${example}" "${ADOLC_DOC_EXAMPLE_SOURCE_DIR}/${example}.cpp")
endforeach()

add_tested_documentation_example(
  abs_normal_struct "${ADOLC_EXAMPLE_SOURCE_DIR}/abs_normal_struct.cpp")
add_tested_documentation_example(
  checkpointing
  "${ADOLC_ADDITIONAL_EXAMPLE_SOURCE_DIR}/checkpointing/checkpointing.cpp")
add_tested_documentation_example(
  edf_object
  "${ADOLC_ADDITIONAL_EXAMPLE_SOURCE_DIR}/ext_diff_func/edfootest.cpp")

set(DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
configure_file("${DOXYFILE_IN}" "${DOXYFILE_OUT}" @ONLY)

# `doxygen` is kept for compatibility with the existing build instructions.
# Both targets compile and run every displayed example before rendering HTML.
add_custom_target(doxygen
  COMMAND ${DOXYGEN_EXECUTABLE} "${DOXYFILE_OUT}"
  DEPENDS ${documentation_example_outputs}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Generating tested ADOL-C API documentation"
  VERBATIM)

add_custom_target(docs DEPENDS doxygen)
