target_sources(adolc PRIVATE
               adalloc.cpp
               adolcerror.cpp
               adouble.cpp
               adouble_tl.cpp
               adouble_tl_hov.cpp
               adouble_tl_indo.cpp
               advector.cpp
               ampisupport.cpp
               ampisupportAdolc.cpp
               checkpointing.cpp
               convolut.cpp
               externfcts.cpp
               externfcts2.cpp
               fixpoint.cpp
               fortutils.cpp
               forward_partx.cpp
               fos_forward.cpp
               fos_pl_forward.cpp
               fos_pl_reverse.cpp
               fos_pl_sig_forward.cpp
               fos_reverse.cpp
               fov_forward.cpp
               fov_offset_forward.cpp
               fov_pl_sig_forward.cpp
               fos_pl_sig_reverse.cpp
               fov_pl_forward.cpp
               fov_reverse.cpp
               hos_forward.cpp
               hos_ov_reverse.cpp
               hos_reverse.cpp
               hov_forward.cpp
               hov_reverse.cpp
               hov_wk_forward.cpp
               indopro_forward_pl.cpp
               indopro_forward_s.cpp
               indopro_forward_t.cpp
               int_forward_s.cpp
               int_forward_t.cpp
               int_reverse_s.cpp
               int_reverse_t.cpp
               interfaces.cpp
               interfacesf.cpp
               medipacksupport.cpp
               nonl_ind_forward_s.cpp
               nonl_ind_forward_t.cpp
               nonl_ind_old_forward_s.cpp
               nonl_ind_old_forward_t.cpp
               pdouble.cpp
               revolve.cpp
               rpl_malloc.cpp
               storemanager.cpp
               tape_interface.cpp
               zos_forward.cpp
               zos_pl_forward.cpp
              )

add_subdirectory(drivers)
add_subdirectory(valuetape)
add_subdirectory(internal)