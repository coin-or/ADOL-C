add_executable(checkpointing checkpointing/checkpointing.cpp)
target_link_libraries(checkpointing PRIVATE adolc)

add_executable(helm_auto_example helm/helm-auto-exam.cpp)
target_link_libraries(helm_auto_example PRIVATE adolc)

if (ENABLE_SPARSE)
    add_executable(sparse_jacobian sparse/sparse_jacobian.cpp)
    target_link_libraries(sparse_jacobian PRIVATE adolc)

    add_executable(sparse_hessian sparse/sparse_hessian.cpp)
    target_link_libraries(sparse_hessian PRIVATE adolc)
endif()
