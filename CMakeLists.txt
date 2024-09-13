cmake_minimum_required(VERSION 3.19)

project(adol-c
  VERSION 2.7.3
  LANGUAGES C CXX
  DESCRIPTION "A Package for Automatic Differentiation of Algorithms Written in C/C++"
  HOMEPAGE_URL "https://github.com/coin-or/ADOL-C")

add_library(adolc SHARED)
add_library(adolc::adolc ALIAS adolc)

target_compile_features(adolc PUBLIC cxx_std_17)

# Make the version of ADOL-C available as compile definitions
target_compile_definitions(adolc PRIVATE
  ADOLC_VERSION=${adol-c_VERSION_MAJOR}
  ADOLC_SUBVERSION=${adol-c_VERSION_MINOR}
  ADOLC_PATCHLEVEL=${adol-c_VERSION_PATCH})

# Set the public include directory containing headers that will be installed
target_include_directories(adolc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/include>
    $<INSTALL_INTERFACE:include>)

# Set an include directory for the internally used library headers.
#
# This includes the files uni5_for.c, fo_rev.c, and ho_rev.c. Even though
# they end with .c, they are used like header files.  Together with some
# preprocessor trickery this is an old-fashioned way to do generic programming.
target_include_directories(adolc
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/src>)


# handle the options
# ------------------

set(UINT_TYPE uint32_t)
set(REAL_TYPE double)

set(ADVBRANCH "#undef ADOLC_ADVANCED_BRANCHING")
set(ADTL_REFCNT "#undef USE_ADTL_REFCOUNTING")
set(SPARSE_DRIVERS "#undef SPARSE_DRIVERS")
set(USE_BOOST_POOL "#undef USE_BOOST_POOL")

# include subdirectories for handling of includes and source files
# ----------------------------------------------------------------

add_subdirectory(ADOL-C)


# export the targets
# ------------------

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(TARGETS adolc EXPORT adolcTargets)
install(EXPORT adolcTargets
  FILE adolc-targets.cmake
  NAMESPACE adolc::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adolc)


# build the ADOL-C C Interface
# ----------------------------

option(CINTERFACE "if the c interface is built" OFF)
if(CINTERFACE)
  add_subdirectory(ADOL-C/c_interface)
endif()