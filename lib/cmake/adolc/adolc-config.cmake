set(ADOLC_VERSION 2.7.3)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was adolc-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(ADOLC_INCLUDE_DIR   "/Users/timsiebert/Projects/adolc_pr/ADOL-C/include" )
set(ADOLC_DATA_DIR      "/Users/timsiebert/Projects/adolc_pr/ADOL-C/" )

include(CMakeFindDependencyMacro)

set(ENABLE_MEDIPACK OFF)
if(ENABLE_MEDIPACK)
  find_dependency(MPI)
  find_dependency(MeDiPack)
endif()

set(ENABLE_OPENMP )
if(ENABLE_OPENMP)
  find_dependency(OpenMP)
endif()

set(WITH_BOOST )
if(WITH_BOOST)
  find_dependency(Boost 1.54 REQUIRED COMPONENTS system)
endif()

# Add the targets file
include("${CMAKE_CURRENT_LIST_DIR}/adolc-targets.cmake")
