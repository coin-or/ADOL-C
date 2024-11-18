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

set(SOURCE_FILES
    adouble.cpp
    main.cpp
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
add_executable(boost-test-adolc ${SOURCE_FILES})
target_include_directories(boost-test-adolc PRIVATE "${ADOLC_INCLUDE_DIR}")
target_link_libraries(boost-test-adolc PRIVATE
    adolc
    Boost::system 
    Boost::unit_test_framework)
