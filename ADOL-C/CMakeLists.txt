add_subdirectory(doc)
add_subdirectory(include)
add_subdirectory(src)
if(ENABLE_DOCEXA OR ENABLE_ADDEXA OR ENABLE_PAREXA)
  add_subdirectory(examples)
endif()

