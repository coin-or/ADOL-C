set(3RDPARTY_BOOST_DIR "" CACHE PATH "The directory where BOOST library is installed")

if(3RDPARTY_BOOST_DIR)
  set(BOOST_ROOT ${3RDPARTY_BOOST_DIR})
endif()

set(BOOST_MIN_VERSION "1.59.0")
set(Boost_NO_BOOST_CMAKE ON)

find_package(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS unit_test_framework system)

if(NOT Boost_FOUND)
  message(FATAL_ERROR "Fatal error: Boost (version >= 1.69.0) required.")
endif()

find_package(OpenMP)

set(SOURCE_FILES
    adouble.cpp
    main.cpp
    pdouble.cpp
    traceCompositeTests.cpp
    tracelessCompositeTests.cpp
    tracelessOperatorScalar.cpp
    tracelessOperatorVector.cpp
    traceOperatorScalar.cpp
    traceOperatorVector.cpp
    traceSecOrderScalar.cpp
    traceSecOrderVector.cpp
    traceFixedPointScalarTests.cpp
    )



# Add all source files from uni5_for
file(GLOB UNI5_FOR_FILES "uni5_for/*.cpp")
list(APPEND SOURCE_FILES ${UNI5_FOR_FILES})

# Add all source files from integration_tests
file(GLOB INTEGRATION_TESTS_FILES "integration_tests/*.cpp")
list(APPEND SOURCE_FILES ${INTEGRATION_TESTS_FILES})

# Add all source files from valuetape
file(GLOB VALUETAPE_FILES "valuetape/*.cpp")
list(APPEND SOURCE_FILES ${VALUETAPE_FILES})

# Add all source files from ho_rev
file(GLOB VALUETAPE_FILES "ho_rev/*.cpp")
list(APPEND SOURCE_FILES ${VALUETAPE_FILES})

add_executable(boost-test-adolc ${SOURCE_FILES})
target_include_directories(boost-test-adolc PRIVATE "${ADOLC_INCLUDE_DIR}")
target_link_libraries(boost-test-adolc PRIVATE
    adolc
    Boost::system 
    Boost::unit_test_framework)


# ADOL-C + OpenMP parallel tests
if(OpenMP_FOUND)
    add_executable(boost-test-adolc-omp openmp.cpp)
    target_include_directories(boost-test-adolc-omp PRIVATE "${ADOLC_INCLUDE_DIR}")
    target_link_libraries(boost-test-adolc-omp PRIVATE
        adolc
        OpenMP::OpenMP_CXX)
endif()
