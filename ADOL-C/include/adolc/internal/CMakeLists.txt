install(FILES
        adubfunc.h
        common.h
        paramfunc.h
        usrparms.h
        DESTINATION "include/adolc/internal")

# Write options into the file adolc_settings.h
configure_file(adolc_settings.h.in ${CMAKE_CURRENT_BINARY_DIR}/adolc_settings.h @ONLY)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/adolc_settings.h
  DESTINATION "include/adolc/internal")
target_include_directories(adolc
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/ADOL-C/include>)
