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
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.31)
        message(FATAL_ERROR "MSVC 19.31 (Visual Studio 2022 v17.1) or higher is required for ADOL-C.")
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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/ADOL-C/include> # for adolc_settings.h if one uses fetchcontent
    $<INSTALL_INTERFACE:include>)

# Set an include directory for the internally used library headers.
#
# This includes the files uni5_for.cpp, fo_rev.cpp, and ho_rev.cpp. Even though
# they end with .cpp, they are used like header files.  Together with some
# preprocessor trickery this is an old-fashioned way to do generic programming.
target_include_directories(adolc
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/src>)


# define options
# --------------

option(ENABLE_MEDIPACK "Build ADOL-C with MeDiPack (MPI) support" FALSE)
option(ENABLE_PYTHON "Build ADOL-C with python/swig support" FALSE)

# The boolean valued comparison operators with two adouble arguments will
# not return boolean results but the active results may be used to automatically
# switch branches in conjunction with condassign or advector (see manual).
option(ENABLE_ADVANCED_BRANCHING "Enable advanced branching operations to reduce retaping" FALSE)

# With this enabled some additional checks will be conducted when setting the
# number of directional derivatives for tapeless numbers using the SetNumDir()
# function.
option(ENABLE_TRACELESS_REFCOUNTING "Enable reference counting for tapeless numbers" FALSE)

# Only the operations involving actual dependency relationships from the
# independent variables will be recorded on the trace, this however
# requires more checks to be performed during the tracing and increases
# tracing time. Useful only if memory is a constraint and tracing is
# done fewer times than derivative computations.
option(ENABLE_ACTIVITY_TRACKING "Enable activity tracking to reduce trace size but increased tracing time" FALSE)

option(ENABLE_HARDDEBUG "Enable ADOL-C hard debug mode" FALSE)
if (ENABLE_HARDDEBUG)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif (ENABLE_HARDDEBUG)

# adouble constructors need to be called. With malloc or realloc that is not the case.
# With this option the adouble can do a late initialization.)
option(ENABLE_LATEINIT "Whether the adouble variables have a late initialize option for functions like malloc/realloc" FALSE)

option(DISABLE_USE_CALLOC "Disable use of calloc and use malloc instead for memory allocation" FALSE)
option(DISABLE_STDCZERO "adouble default constructor does not initialize the value to zero" FALSE)

option(ENABLE_BOOST_POOL "Enable the use of boost pool." FALSE)

option(ENABLE_SPARSE "NOT SUPPORTED!: Build with Colpack to enable sparse AD.")

# build the c interface
option(BUILD_INTERFACE OFF)

# build the tests
option(BUILD_TESTS OFF)

# handle the options
# ------------------

# Whether to build ADOL-C with adjoinable MPI (AMPI) or MeDiPack (MPI) support
target_compile_definitions(adolc PRIVATE
  $<$<BOOL:${ENABLE_MEDIPACK}>:ADOLC_MEDIPACK_SUPPORT=1>)

if(ENABLE_MEDIPACK)
  find_package(MPI REQUIRED)
  find_package(MeDiPack 1.0...<1.2 REQUIRED)                  # cmake >= 3.19
  target_link_libraries(adolc PUBLIC MeDiPack MPI::MPI_CXX)
endif()

# enable python and swig
target_compile_definitions(adolc PRIVATE
  $<$<BOOL:${ENABLE_PYTHON}>:ADOLC_python_SUPPORT=1>)

if(ENABLE_PYTHON)
  find_package(Python3 3.8 REQUIRED)
  find_package(SWIG REQUIRED COMPONENTS python)
endif()

if (ENABLE_BOOST_POOL)
  find_package(Boost 1.54 REQUIRED COMPONENTS system)
  target_link_libraries(adolc PUBLIC Boost::boost Boost::system)
endif()

if (ENABLE_SPARSE)
  # TODO add colpack to 
endif()


# Set the macros

target_compile_definitions(adolc PRIVATE
  $<$<BOOL:${ENABLE_ACTIVITY_TRACKING}>:ADOLC_TRACK_ACTIVITY=1>
  $<$<BOOL:${ENABLE_LATEINIT}>:ADOLC_ADOUBLE_LATEINIT=1>
  $<$<BOOL:${ENABLE_ADVANCED_BRANCHING}>:ADOLC_ADVANCED_BRANCHING=1>
  $<$<BOOL:${ENABLE_TRACELESS_REFCOUNTING}>:USE_ADTL_REFCOUNTING=1>
  $<$<BOOL:${WITH_BOOST}>:ENABLE_BOOST_POOL=1>
  $<$<NOT:$<BOOL:${DISABLE_STDCZERO}>>:ADOLC_ADOUBLE_STDCZERO=1>)


# Debug mode enabled
target_compile_definitions(adolc PRIVATE
  $<$<OR:$<CONFIG:Debug>,$<BOOL:${ENABLE_HARDDEBUG}>>:ADOLC_DEBUG=1>
  $<$<BOOL:${ENABLE_HARDDEBUG}>:ADOLC_HARDDEBUG=1>)


# include subdirectories for handling of includes and source files
# ----------------------------------------------------------------

add_subdirectory(ADOL-C)


# include the c interface of adolc
# ----------------------------------------------------------------
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

write_basic_package_version_file(adolc-config-version.cmake
  VERSION ${adol-c_VERSION}
  COMPATIBILITY SameMinorVersion)               # cmake >= 3.11

configure_package_config_file(adolc-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/adolc-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adolc)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/adolc-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/adolc-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adolc)

export(TARGETS adolc NAMESPACE adolc:: FILE adolc-targets.cmake)

# print a summary of found packages and set options
# -------------------------------------------------

include(FeatureSummary)
feature_summary(WHAT ALL)
