add_executable(fminmax_test fminmax.cpp)
target_include_directories(fminmax_test PRIVATE "${ADOLC_INCLUDE_DIR}")
target_link_libraries(fminmax_test PRIVATE adolc)
add_test(NAME fminmax COMMAND fminmax_test)

add_executable(powexam_test powexam.cpp)
target_include_directories(powexam_test PRIVATE "${ADOLC_INCLUDE_DIR}")
target_link_libraries(powexam_test PRIVATE adolc)
add_test(NAME powexam COMMAND powexam_test)

add_executable(speelpenning_test speelpenning.cpp)
target_include_directories(speelpenning_test PRIVATE "${ADOLC_INCLUDE_DIR}")
target_link_libraries(speelpenning_test PRIVATE adolc)
add_test(NAME speelpenning COMMAND speelpenning_test)

add_executable(callback_test callback.cpp)
target_include_directories(callback_test PRIVATE "${ADOLC_INCLUDE_DIR}")
target_link_libraries(callback_test PRIVATE adolc)
add_test(NAME callback COMMAND callback_test)
