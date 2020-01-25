cmake_minimum_required(VERSION 3.0)
project(boost-test-adolc CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

## BOOST
set(3RDPARTY_BOOST_DIR "" CACHE PATH "The directory where BOOST library is installed")

if(3RDPARTY_BOOST_DIR)
  set(BOOST_ROOT ${3RDPARTY_BOOST_DIR})
#  set(BOOST_INCLUDEDIR ${3RDPARTY_BOOST_DIR}/include)
#  set(BOOST_LIBRARYDIR ${3RDPARTY_BOOST_DIR}/lib)
endif()

set(BOOST_MIN_VERSION "1.59.0")
set(Boost_NO_BOOST_CMAKE ON)

find_package(Boost ${BOOST_MIN_VERSION} REQUIRED)

if(NOT Boost_FOUND)
  message(FATAL_ERROR "Fatal error: Boost (version >= 1.69.0) required.")
else()
  message(STATUS "Setting up BOOST")
  message(STATUS "Boost include: " ${Boost_INCLUDE_DIRS})
  message(STATUS "Boost library: " ${Boost_LIBRARY_DIRS})
endif()

## ADOL-C
set (ADOLC_BASE "" CACHE PATH "The directory where ADOL-C is installed")
if(ADOLC_BASE)
  message(STATUS "Setting up ADOL-C")

  unset(ADOLC_INCLUDE_DIR CACHE)
  find_path(ADOLC_INCLUDE_DIR NAMES adolc/adolc.h PATHS ${ADOLC_BASE}/include NO_DEFAULT_PATH)
  if(NOT ADOLC_INCLUDE_DIR)
    message(FATAL_ERROR "Fatal error: ADOL-C include directory not found, check if ADOLC_BASE path is correct")
  endif()
  
  unset(ADOLC_LIBRARY CACHE)
  find_library(ADOLC_LIBRARY NAMES adolc PATHS ${ADOLC_BASE}/lib64 ${ADOLC_BASE}/lib NO_DEFAULT_PATHS)
  if(NOT ADOLC_LIBRARY)
    message(FATAL_ERROR "Fatal error: ADOL-C library not found, check if ADOLC_BASE path is correct")
  endif()
  
  unset(ADOLC_LIBRARY_DIR CACHE)
  get_filename_component(ADOLC_LIBRARY_DIR ${ADOLC_LIBRARY} DIRECTORY CACHE) 
  
  message(STATUS "ADOL-C include: " ${ADOLC_INCLUDE_DIR})
  message(STATUS "ADOL-C library: " ${ADOLC_LIBRARY_DIR})
else()
  message(FATAL_ERROR "ADOLC_BASE directory has to be specified")
endif()

include_directories(${Boost_INCLUDE_DIRS} ${ADOLC_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS} ${ADOLC_LIBRARY_DIR})

set(SOURCE_FILES main.cpp tracelessOperatorScalar.cpp tracelessOperatorVector.cpp tracelessCompositeTests.cpp traceOperatorScalar.cpp traceOperatorVector.cpp traceCompositeTests.cpp traceSecOrderScalar.cpp traceSecOrderVector.cpp)
add_executable(boost-test-adolc ${SOURCE_FILES})

target_link_libraries(boost-test-adolc -ladolc -lboost_system -lboost_unit_test_framework)
