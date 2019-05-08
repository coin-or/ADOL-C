cmake_minimum_required(VERSION 3.0)
project(boost-test-adolc CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

## ADOL-C
include_directories(/home/banovicm/adolc_base_boost/include /home/banovicm/thirdparty/boost/boost_1_69_0_install/include)
link_directories(/home/banovicm/adolc_base_boost/lib64 /home/banovicm/thirdparty/boost/boost_1_69_0_install/lib)


set(SOURCE_FILES main.cpp tracelessOperatorScalar.cpp tracelessOperatorVector.cpp traceOperatorScalar.cpp traceOperatorVector.cpp)
add_executable(boost-test-adolc ${SOURCE_FILES})

target_link_libraries(boost-test-adolc -ladolc -lboost_system -lboost_unit_test_framework)
