add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(doc)

if(ENABLE_DOCEXA OR ENABLE_ADDEXA OR ENABLE_PAREXA)
  add_subdirectory(examples)
endif()