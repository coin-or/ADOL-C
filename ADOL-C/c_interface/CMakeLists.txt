cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(ExternalADOLC SHARED IMPORTED)
set_target_properties(ExternalADOLC PROPERTIES
    IMPORTED_LOCATION ${ADOLC_INSTALL_DIR}/lib/libadolc.dylib
    INTERFACE_INCLUDE_DIRECTORIES ${ADOLC_INSTALL_DIR}/include
)
add_library(ADOLCInterface SHARED ADOLC_TB_interface.cpp ADOLC_TL_interface.cpp array_handler.cpp driver_interface.cpp)
target_link_libraries(ADOLCInterface ExternalADOLC) 

