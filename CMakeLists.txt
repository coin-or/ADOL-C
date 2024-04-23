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

# Handle single .cpp files directly in the examples directory
file(GLOB single_sources "ADOL-C/examples/*.cpp")
# duplicate symbol getOneADValue in traceless_higher_order and libadolc.a so skip
list(REMOVE_ITEM single_sources "${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/examples/traceless_higher_order.cpp")

foreach(source_file ${single_sources})
  get_filename_component(base_name ${source_file} NAME_WE)
  string(REGEX REPLACE "[^a-zA-Z0-9]" "_" base_name ${base_name})
  add_executable(${base_name} ${source_file})
  target_link_libraries(${base_name} adolc::adolc)
endforeach()

# Handle .cpp files in subdirectories under additional_examples
file(GLOB additional_sources "ADOL-C/examples/additional_examples/*/")
list(REMOVE_ITEM additional_sources "${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/examples/additional_examples/clock")
# build clock as library
add_library(clock "${CMAKE_SOURCE_DIR}/ADOL-C/examples/additional_examples/clock/myclock.cpp")
target_include_directories(clock PUBLIC "${CMAKE_SOURCE_DIR}/ADOL-C/examples/additional_examples/clock")


# remove ipopt/
# if not defined ADOLC_WITH_IPOPT
if(NOT ADOLC_WITH_IPOPT)
  list(REMOVE_ITEM additional_sources "${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/examples/additional_examples/ipopt")
endif()
if(NOT ADOLC_WITH_CUDA)
  list(REMOVE_ITEM additional_sources "${CMAKE_CURRENT_SOURCE_DIR}/ADOL-C/examples/additional_examples/cuda")
endif()


# ideally each example directory has a CMakelists.txt file that specifies how to build the example
foreach(source_file ${additional_sources})
  # test if source_file is a directory
  if(NOT IS_DIRECTORY ${source_file})
    continue()
  endif()
  get_filename_component(dir_name ${source_file} NAME_WE)
  file(GLOB sources_in_dir "${source_file}/*.cpp")


  # if we're in ADOL-C/examples/additional_examples/lufact/ then compile LU.cpp as a
  # library and the other two .cpp as executables
  if(${dir_name} STREQUAL "lufact")
    add_library(LU "${source_file}/LU.cpp")
    target_link_libraries(LU adolc::adolc)
    # remove LU.cpp from sources_in_dir
    list(REMOVE_ITEM sources_in_dir "${source_file}/LU.cpp")
    foreach(source_in_dir ${sources_in_dir})
      get_filename_component(base_name ${source_in_dir} NAME_WE)
      string(REGEX REPLACE "[^a-zA-Z0-9]" "_" base_name ${base_name})
      set(target_name "additional_${base_name}")
      add_executable(${target_name} ${source_in_dir})
      target_link_libraries(${target_name} adolc::adolc LU)
    endforeach()
    continue()
  endif()

  # if we're in ADOL-C/examples/additional_examples/ext_diff_func/ then compile
  # each cpp as its own executable
  set(compile_separate "detexam" "ext_diff_func" "helm" "lighthouse" "openmp_exam" "sparse" "tapesave" "taylor")
  if(${dir_name} IN_LIST compile_separate)
    foreach(source_in_dir ${sources_in_dir})
      get_filename_component(base_name ${source_in_dir} NAME_WE)
      string(REGEX REPLACE "[^a-zA-Z0-9]" "_" base_name ${base_name})
      set(target_name "additional_${base_name}")
      add_executable(${target_name} ${source_in_dir})
      target_link_libraries(${target_name} adolc::adolc clock)
    endforeach()
    continue()
  endif()

  # if we're in ode then compile vgenodemain.cpp 3 times for each of the other
  # *.cpp files and use their name as target
  if(${dir_name} STREQUAL "ode")
    list(REMOVE_ITEM sources_in_dir "${source_file}/vgenodemain.cpp")
    foreach(source_in_dir ${sources_in_dir})
      get_filename_component(base_name ${source_in_dir} NAME_WE)
      string(REGEX REPLACE "[^a-zA-Z0-9]" "_" base_name ${base_name})
      set(target_name "additional_${base_name}")
      add_executable(${target_name} "${source_file}/vgenodemain.cpp" ${source_in_dir})
      target_link_libraries(${target_name} adolc::adolc clock)
    endforeach()
    continue()
  endif()

  # if we're in ode then compile vgenodemain.cpp 3 times for each of the other
  # *.cpp files and use their name as target
  if(${dir_name} STREQUAL "ode")
    list(REMOVE_ITEM sources_in_dir "${source_file}/vgenodemain.cpp")
    foreach(source_in_dir ${sources_in_dir})
      get_filename_component(base_name ${source_in_dir} NAME_WE)
      string(REGEX REPLACE "[^a-zA-Z0-9]" "_" base_name ${base_name})
      set(target_name "additional_${base_name}")
      add_executable(${target_name} "${source_file}/vgenodemain.cpp" ${source_in_dir})
      target_link_libraries(${target_name} adolc::adolc clock)
    endforeach()
    continue()
  endif()

  # if we're in timing then compile vgenmain for every vfunc_*.cpp file and sgenmain for every sfunc_*.cpp file
  if(${dir_name} STREQUAL "timing")
    # compile rotations.cpp as a library
    add_library(rotations "${source_file}/rotations.cpp")
    target_link_libraries(rotations adolc::adolc)

    list(REMOVE_ITEM sources_in_dir "${source_file}/vgenmain.cpp")
    list(REMOVE_ITEM sources_in_dir "${source_file}/sgenmain.cpp")
    foreach(source_in_dir ${sources_in_dir})
      get_filename_component(base_name ${source_in_dir} NAME_WE)
      string(REGEX REPLACE "[^a-zA-Z0-9]" "_" base_name ${base_name})
      if(${base_name} MATCHES "vfunc_.*")
        set(target_name "additional_vgenmain_${base_name}")
        add_executable(${target_name} "${source_file}/vgenmain.cpp" ${source_in_dir})
      elseif(${base_name} MATCHES "sfunc_.*")
        set(target_name "additional_sgenmain_${base_name}")
        add_executable(${target_name} "${source_file}/sgenmain.cpp" ${source_in_dir})
      endif()
      target_link_libraries(${target_name} adolc::adolc clock rotations)
    endforeach()
    continue()
  endif()

  set(target_name "additional_${dir_name}")
  add_executable(${target_name} ${sources_in_dir})
  target_link_libraries(${target_name} adolc::adolc clock)
endforeach()
