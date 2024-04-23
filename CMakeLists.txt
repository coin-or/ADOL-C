cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(ADOLC VERSION 1.0 LANGUAGES C CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# boost
find_package(boost REQUIRED)
message(STATUS "Boost include dirs: ${boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${boost_LIBRARIES}")

# openmp
if(APPLE)
  # Assuming libomp is installed
  set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
endif()
find_package(OpenMP REQUIRED)

# ADOL-C/include/adolc/internal/adolc_settings.h.in → ADOL-C/include/adolc/internal/adolc_settings.h
function(configure_adolc_settings)
    # Set default values for the variables if not already set
    set(UINT_TYPE "uint32_t" CACHE STRING "Type for locint")
    set(REAL_TYPE "double" CACHE STRING "Type for revreal")
    set(ATRIG_ERF "OFF" CACHE BOOL "Enable asinh, acosh, atanh, erf")
    set(ADVBRANCH "OFF" CACHE BOOL "Enable advanced branching")
    set(ADTL_REFCNT "OFF" CACHE BOOL "Use reference counting for tapeless numbers")
    set(SPARSE_DRIVERS "ON" CACHE BOOL "Sparse drivers have been compiled")
    set(USE_BOOST_POOL "OFF" CACHE BOOL "Use Boost Library Pool allocator")

    # Convert boolean values to C preprocessor directives
    if(${ATRIG_ERF})
        set(ATRIG_ERF "#define ATRIG_ERF")
    else()
        set(ATRIG_ERF "#undef ATRIG_ERF")
    endif()

    if(${ADVBRANCH})
        set(ADVBRANCH "#define ADOLC_ADVANCED_BRANCHING")
    else()
        set(ADVBRANCH "#undef ADOLC_ADVANCED_BRANCHING")
    endif()

    if(${ADTL_REFCNT})
        set(ADTL_REFCNT "#define USE_ADTL_REFCOUNTING")
    else()
        set(ADTL_REFCNT "#undef USE_ADTL_REFCOUNTING")
    endif()

    if(${SPARSE_DRIVERS})
        set(SPARSE_DRIVERS "#define SPARSE_DRIVERS 1")
    else()
        set(SPARSE_DRIVERS "#undef SPARSE_DRIVERS")
    endif()

    if(${USE_BOOST_POOL})
        set(USE_BOOST_POOL "#define USE_BOOST_POOL 1")
    else()
        set(USE_BOOST_POOL "#define USE_BOOST_POOL 0")
    endif()

    # Configure the file
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/include/adolc/internal/adolc_settings.h.in
                   ${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/include/adolc/internal/adolc_settings.h)
endfunction()
configure_adolc_settings()

# Gather all C and C++ source files
file(GLOB_RECURSE ADOLC_SOURCES "ADOL-C/src/*.cpp" "ADOL-C/src/*.c")
# remove file ADOL-C/src/fo_rev.c
list(REMOVE_ITEM ADOLC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/src/fo_rev.c")
list(REMOVE_ITEM ADOLC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/src/ho_rev.c")
list(REMOVE_ITEM ADOLC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/src/uni5_for.c")

# Create static library
add_library(adolc STATIC ${ADOLC_SOURCES})

# Assuming we're using header-only parts of Boost, we just need to include directories
target_link_libraries(adolc PUBLIC Boost::boost OpenMP::OpenMP_CXX)
target_compile_definitions(adolc PRIVATE -DADOLC_VERSION=2)
target_compile_definitions(adolc PRIVATE -DADOLC_SUBVERSION=7)
target_compile_definitions(adolc PRIVATE -DADOLC_PATCHLEVEL=2)
target_compile_definitions(adolc PRIVATE -DADOLC_USE_CALLOC)
target_include_directories(adolc PRIVATE ADOL-C/src)
target_include_directories(adolc PUBLIC ADOL-C/include)
# make into adolc::adolc
add_library(adolc::adolc ALIAS adolc)

add_executable(detexam ADOL-C/examples/detexam.cpp)
target_link_libraries(detexam adolc::adolc)
