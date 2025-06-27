#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "adolc::adolc" for configuration "Debug"
set_property(TARGET adolc::adolc APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(adolc::adolc PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libadolc.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libadolc.dylib"
  )

list(APPEND _cmake_import_check_targets adolc::adolc )
list(APPEND _cmake_import_check_files_for_adolc::adolc "${_IMPORT_PREFIX}/lib/libadolc.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
