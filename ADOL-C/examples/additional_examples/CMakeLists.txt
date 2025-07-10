add_executable(checkpointing checkpointing/checkpointing.cpp)
target_link_libraries(checkpointing PRIVATE adolc)


add_executable(helm_auto_example helm/helm-auto-exam.cpp)
target_link_libraries(helm_auto_example PRIVATE adolc)