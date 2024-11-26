add_executable(detexam detexam.cpp)
target_link_libraries(detexam PRIVATE adolc)

add_executable(luexam luexam.cpp)
target_link_libraries(luexam PRIVATE adolc)

add_executable(odexam odexam.cpp)
target_link_libraries(odexam PRIVATE adolc)

add_executable(powexam powexam.cpp)
target_link_libraries(powexam PRIVATE adolc)

add_executable(speelpenning speelpenning.cpp)
target_link_libraries(speelpenning PRIVATE adolc)

add_executable(traceless_scalar traceless_scalar.cpp)
target_link_libraries(traceless_scalar PRIVATE adolc)

add_executable(traceless_vector traceless_vector.cpp)
target_link_libraries(traceless_vector PRIVATE adolc)

add_executable(traceless_higher_order traceless_higher_order.cpp)
target_link_libraries(traceless_higher_order PRIVATE adolc)

if(ENABLE_SPARSE)
  add_executable(traceless_vector_indo traceless_vector_indo.cpp)
  target_link_libraries(traceless_vector_indo PRIVATE adolc)
endif()

if(ENABLE_ADDEXA OR ENABLE_PAREXA)
  # add_subdirectory(additional_examples) # TODO
endif()