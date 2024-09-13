cmake_minimum_required(VERSION 3.27)

project(TestADOLC VERSION 1.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(ExternalADOLC SHARED IMPORTED)
set_target_properties(ExternalADOLC PROPERTIES
    IMPORTED_LOCATION "${CMAKE_INSTALL_PREFIX}/lib/libadolc.dylib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/include"
)
add_library(ADOLCInterface SHARED ADOLCTBInterface.cpp ADOLCTLInterface.cpp array_handler.cpp)
target_link_libraries(ADOLCInterface ExternalADOLC) 
