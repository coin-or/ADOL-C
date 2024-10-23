add_library(ADOLCInterface SHARED ADOLC_TB_interface.cpp ADOLC_TL_interface.cpp array_handler.cpp driver_interface.cpp)
target_link_libraries(ADOLCInterface adolc) 
