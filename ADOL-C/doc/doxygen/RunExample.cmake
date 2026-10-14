if(NOT DEFINED EXAMPLE_EXECUTABLE
   OR NOT DEFINED OUTPUT_FILE
   OR NOT DEFINED RUNTIME_DIRECTORY)
  message(FATAL_ERROR
    "EXAMPLE_EXECUTABLE, OUTPUT_FILE, and RUNTIME_DIRECTORY are required")
endif()

file(MAKE_DIRECTORY "${RUNTIME_DIRECTORY}")
execute_process(
  COMMAND "${EXAMPLE_EXECUTABLE}"
  WORKING_DIRECTORY "${RUNTIME_DIRECTORY}"
  RESULT_VARIABLE example_result
  OUTPUT_VARIABLE example_output
  ERROR_VARIABLE example_error
  TIMEOUT 60)

if(NOT example_result STREQUAL "0")
  message(FATAL_ERROR
    "Documentation example failed with exit code ${example_result}:\n"
    "${example_output}${example_error}")
endif()

get_filename_component(output_directory "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
file(WRITE "${OUTPUT_FILE}" "${example_output}")
