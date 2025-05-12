cmake_minimum_required(VERSION 3.19)

project(adol-c
  VERSION 2.7.3
  LANGUAGES C CXX
  DESCRIPTION "A Package for Automatic Differentiation of Algorithms Written in C/C++"
  HOMEPAGE_URL "https://github.com/coin-or/ADOL-C")

add_library(adolc SHARED)
add_library(adolc::adolc ALIAS adolc)

target_compile_features(adolc PUBLIC cxx_std_20)

cmake_minimum_required(VERSION 3.15)
project(MyProject CXX)

# Enforce C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # only standard c++

# Check for minimum compiler versions that support most of C++20
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
        message(FATAL_ERROR "GCC 11 or higher is required for ADOL-C.")
    endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
        message(FATAL_ERROR "Clang 13 or higher is required for ADOL-C.")
    endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 1931)
        message(FATAL_ERROR "MSVC 19.31 (Visual Studio 2019 v16.8) or higher is required for ADOL-C.")
    endif()

else()
    message(WARNING "Unknown compiler — C++20 compatibility not guaranteed.")
endif()


# Make the version of ADOL-C available as compile definitions
target_compile_definitions(adolc PRIVATE
  ADOLC_VERSION=${adol-c_VERSION_MAJOR}
  ADOLC_SUBVERSION=${adol-c_VERSION_MINOR}
  ADOLC_PATCHLEVEL=${adol-c_VERSION_PATCH})


# win specific flags
if (WIN32)
  target_compile_definitions(adolc PRIVATE     
      # handles __declspec
      ADOLC_DLL
  )   
endif(WIN32) 


# Set the public include directory containing headers that will be installed
target_include_directories(adolc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/include>
    $<INSTALL_INTERFACE:include>)

# Set an include directory for the internally used library headers.
#
# This includes the files uni5_for.cpp, fo_rev.cpp, and ho_rev.cpp. Even though
# they end with .cpp, they are used like header files.  Together with some
# preprocessor trickery this is an old-fashioned way to do generic programming.
target_include_directories(adolc
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/src>)


# handle the options
# ------------------

set(UINT_TYPE size_t)
set(REAL_TYPE double)

set(ADVBRANCH "#undef ADOLC_ADVANCED_BRANCHING")
set(ADTL_REFCNT "#undef USE_ADTL_REFCOUNTING")
set(SPARSE_DRIVERS "#undef SPARSE_DRIVERS")

option(ENABLE_BOOST_POOL "Flag to activate boost-pool support" OFF)
if(ENABLE_BOOST_POOL)
  message(STATUS "Boost-pool support is enabled.")
  set(USE_BOOST_POOL "#define USE_BOOST_POOL 1")
else()
  message(STATUS "Boost-pool support is disabled.")
  set(USE_BOOST_POOL "#undef USE_BOOST_POOL")
endif()


# include subdirectories for handling of includes and source files
# ----------------------------------------------------------------

add_subdirectory(ADOL-C)


# include the c interface of adolc
# ----------------------------------------------------------------
option(BUILD_INTERFACE OFF)
if(BUILD_INTERFACE)
    # set rpath for execution to the install location of adolc
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    add_subdirectory(ADOL-C/c_interface)
    install(TARGETS ADOLCInterface EXPORT ADOLCInterfaceTargets)
    install(EXPORT ADOLCInterfaceTargets 
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adolc)
endif()

# build the tests 
# ------------------------------------
option(BUILD_TESTS OFF)
if(BUILD_TESTS)
    set(ADOLC_INCLUDE_DIR "${CMAKE_BINARY_DIR}/ADOL-C/include")
    add_subdirectory(ADOL-C/boost-test)

    enable_testing()
    add_test(NAME boost-test-adolc
      COMMAND boost-test-adolc)
endif()

# build the adolc and tests with coverage
# ------------------------------------
option(BUILD_TESTS_WITH_COV OFF)
if(BUILD_TESTS_WITH_COV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g --coverage")

    set(ADOLC_INCLUDE_DIR "${CMAKE_BINARY_DIR}/ADOL-C/include")
    add_subdirectory(ADOL-C/boost-test)

    enable_testing()
    add_test(NAME boost-test-adolc
      COMMAND boost-test-adolc)
endif()


# export the targets
# ------------------

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(TARGETS adolc EXPORT adolcTargets)
install(EXPORT adolcTargets
  FILE adolc-targets.cmake
  NAMESPACE adolc::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adolc)




