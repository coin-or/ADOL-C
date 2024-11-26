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
# This includes the files uni5_for.c, fo_rev.c, and ho_rev.c. Even though
# they end with .c, they are used like header files.  Together with some
# preprocessor trickery this is an old-fashioned way to do generic programming.
target_include_directories(adolc
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/src>)


# define options
# --------------

include(FeatureSummary)

# Whether to build ADOL-C with MeDiPack (MPI) support
option(ENABLE_MEDIPACK "Build ADOL-C with MeDiPack (MPI) support" FALSE)
add_feature_info(ENABLE_MEDIPACK ENABLE_MEDIPACK "MeDiPack: MPI wrapper for Algorithmic Differentiation tools.")
target_compile_definitions(adolc PRIVATE
  $<$<BOOL:${ENABLE_MEDIPACK}>:ADOLC_MEDIPACK_SUPPORT=1>)
if(ENABLE_MEDIPACK)
  find_package(MPI REQUIRED)
  find_package(MeDiPack 1.0...<1.2 REQUIRED)                  # cmake >= 3.19
  target_link_libraries(adolc PUBLIC MeDiPack MPI::MPI_CXX)
endif()

# Whether to use calloc or malloc for memory allocation
option(DISABLE_USE_CALLOC "Disable use of calloc and use malloc instead for memory allocation" FALSE)
add_feature_info(DISABLE_USE_CALLOC DISABLE_USE_CALLOC "Use malloc instead of calloc for memory allocation.")
target_compile_definitions(adolc PRIVATE
  $<$<NOT:$<BOOL:${DISABLE_USE_CALLOC}>>:ADOLC_USE_CALLOC=1>)

# Whether to use 32-bit or 64-bit locations
option(ENABLE_ULONG "Enable 64-bit locations (only available on 64-bit systems)" FALSE)
add_feature_info(ENABLE_ULONG ENABLE_ULONG "Use 64-bit unsigned integers to determine how many adoubles can be simultaneously alive.")
if(ENABLE_ULONG)
  set(UINT_TYPE uint64_t)
else()
  set(UINT_TYPE uint32_t)
endif()

# Whether to use single or double precision
option(DISABLE_DOUBLE "Disable double precision arithmetic (untested)" FALSE)
add_feature_info(DISABLE_DOUBLE DISABLE_DOUBLE "Use float instead of double precision arithmetic.")
if (DISABLE_DOUBLE)
  message(WARNING "using 'float' is not well tested, please report bugs if you find any...")
  set(REAL_TYPE float)
else()
  set(REAL_TYPE double)
endif()

# The boolean valued comparison operators with two adouble arguments will
# not return boolean results but the active results may be used to automatically
# switch branches in conjunction with condassign or advector (see manual).
option(ENABLE_ADVANCED_BRANCHING "Enable advanced branching operations to reduce retaping" FALSE)
add_feature_info(ENABLE_ADVANCED_BRANCHING ENABLE_ADVANCED_BRANCHING "Advanced branching operations to reduce retaping.")
if(ENABLE_ADVANCED_BRANCHING)
  set(ADVBRANCH "#define ADOLC_ADVANCED_BRANCHING 1")
else()
  set(ADVBRANCH "#undef ADOLC_ADVANCED_BRANCHING")
endif()

# With this enabled some additional checks will be conducted when setting the
# number of directional derivatives for tapeless numbers using the SetNumDir()
# function.
option(ENABLE_TRACELESS_REFCOUNTING "Enable reference counting for tapeless numbers" FALSE)
add_feature_info(ENABLE_TRACELESS_REFCOUNTING ENABLE_TRACELESS_REFCOUNTING "Reference counting for tapeless numbers.")
if(ENABLE_TRACELESS_REFCOUNTING)
  set(ADTL_REFCNT "#define USE_ADTL_REFCOUNTING 1")
else()
  set(ADTL_REFCNT "#undef USE_ADTL_REFCOUNTING")
endif()

# Only the operations involving actual dependency relationships from the
# independent variables will be recorded on the trace, this however
# requires more checks to be performed during the tracing and increases
# tracing time. Useful only if memory is a constraint and tracing is
# done fewer times than derivative computations.
option(ENABLE_ACTIVITY_TRACKING "Enable activity tracking to reduce trace size but increased tracing time" FALSE)
add_feature_info(ENABLE_ACTIVITY_TRACKING ENABLE_ACTIVITY_TRACKING "Activity tracking to reduce trace size but increased tracing time.")
target_compile_definitions(adolc PRIVATE
  $<$<BOOL:${ENABLE_ACTIVITY_TRACKING}>:ADOLC_TRACK_ACTIVITY=1>)

# Whether to use ADOL-C hard debug mode
option(ENABLE_HARDDEBUG "Enable ADOL-C hard debug mode" FALSE)
add_feature_info(ENABLE_HARDDEBUG ENABLE_HARDDEBUG "Hard debug mode.")
if (ENABLE_HARDDEBUG)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif (ENABLE_HARDDEBUG)
target_compile_definitions(adolc PRIVATE
  $<$<OR:$<CONFIG:Debug>,$<BOOL:${ENABLE_HARDDEBUG}>>:ADOLC_DEBUG=1>
  $<$<BOOL:${ENABLE_HARDDEBUG}>:ADOLC_HARDDEBUG=1>)

# The adouble default constructor does not initialize the value to zero
# (improves performance but yields incorrect results for implicit array initializations, see manual)
option(DISABLE_STDCZERO "adouble default constructor does not initialize the value to zero" FALSE)
add_feature_info(DISABLE_STDCZERO DISABLE_STDCZERO "Do not initialize the values to zero.")
target_compile_definitions(adolc PRIVATE
  $<$<NOT:$<BOOL:${DISABLE_STDCZERO}>>:ADOLC_ADOUBLE_STDCZERO=1>)

# adouble constructors need to be called. With malloc or realloc that is not the case.
# With this option the adouble can do a late initialization.)
option(ENABLE_LATEINIT "Whether the adouble variables have a late initialize option for functions like malloc/realloc" FALSE)
add_feature_info(ENABLE_LATEINIT ENABLE_LATEINIT "adouble can do a late initialization.")
target_compile_definitions(adolc PRIVATE
  $<$<BOOL:${ENABLE_LATEINIT}>:ADOLC_ADOUBLE_LATEINIT=1>)

# Whether errno is thread save
option(ENABLE_TSERRNO "Use errno as thread number cache" FALSE)
add_feature_info(ENABLE_TSERRNO ENABLE_TSERRNO "Use errno as thread number cache.")
target_compile_definitions(adolc PRIVATE
  $<$<BOOL:${ENABLE_TSERRNO}>:ADOLC_THREADSAVE_ERRNO=1>)

# Whether to enable OpenMP
option(ENABLE_OPENMP "Enable OpenMP" FALSE)
add_feature_info(ENABLE_OPENMP ENABLE_OPENMP "Enable OpenMP.")
if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(adolc PUBLIC OpenMP::OpenMP_CXX)
endif()


find_package(Boost 1.54 COMPONENTS system)
set(WITH_BOOST 0)
if(Boost_FOUND)
  target_link_libraries(adolc PUBLIC Boost::boost Boost::system)
  set(WITH_BOOST 1)
  # TODO: check for header boost/pool/pool_alloc.hpp
  set(USE_BOOST_POOL "#define USE_BOOST_POOL 1")
  if(NOT ENABLE_OPENMP)
    # Boost pool should not assume multithreading
    target_compile_definitions(adolc PRIVATE BOOST_POOL_NO_MT=1)
  endif()
endif()

# Whether to build sparse drivers
option(ENABLE_SPARSE "build sparse drivers" FALSE)
add_feature_info(ENABLE_SPARSE ENABLE_SPARSE "build sparse drivers")
set(SPARSE_DRIVERS "#undef SPARSE_DRIVERS")
if(ENABLE_SPARSE)
  set(SPARSE_DRIVERS "#define SPARSE_DRIVERS 1")
  target_compile_definitions(adolc PRIVATE SPARSE=1)
endif()

# Whether to build documented examples
option(ENABLE_DOCEXA "build documented examples" FALSE)
add_feature_info(ENABLE_DOCEXA ENABLE_DOCEXA "build documented examples")

# Whether to build additional examples
option(ENABLE_ADDEXA "build additional examples" FALSE)
add_feature_info(ENABLE_ADDEXA ENABLE_ADDEXA "build additional examples")

# Whether to build parallel example
include(CMakeDependentOption)
cmake_dependent_option(ENABLE_PAREXA "build parallel example" FALSE ENABLE_OPENMP FALSE)
add_feature_info(ENABLE_PAREXA ENABLE_PAREXA "build parallel example")

# Whether tape_doc should compute values as it prints the tape contents
option(DISABLE_TAPEDOC_VALUES "should the tape_doc routine compute the values as it interprets and prints the tape contents" TRUE)
add_feature_info(DISABLE_TAPEDOC_VALUES DISABLE_TAPEDOC_VALUES "should the tape_doc routine compute the values as it interprets and prints the tape contents")
target_compile_definitions(adolc PRIVATE
  $<$<NOT:$<BOOL:${DISABLE_TAPEDOC_VALUES}>>:ADOLC_TAPE_DOC_VALUES=1>)


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
