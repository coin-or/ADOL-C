##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id$
##
## Copyright (C) Andrea Walther, Andreas Kowarz
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

AM_CFLAGS            = @ac_adolc_cflags@ -DADOLC_INTERNAL=1 -std=gnu99
AM_CXXFLAGS          = @ac_adolc_cxxflags@ -DADOLC_INTERNAL=1

AM_CPPFLAGS          = -I$(top_builddir)/ADOL-C/include -I$(top_srcdir)/ADOL-C/include -I$(top_srcdir)/ADOL-C/src

if SPARSE
SUBDIRS              = drivers sparse tapedoc lie
else
SUBDIRS              = drivers tapedoc lie
endif

MAINTAINERCLEANFILES = Makefile.in *~ *.orig adolc_config.h config.h.in

EXTRA_DIST           = uni5_for.c fo_rev.c ho_rev.c

noinst_LTLIBRARIES      = libadolcsrc.la
libadolcsrc_la_LDFLAGS  =

if BUILD_ADOLC_AMPI_SUPPORT
noinst_LTLIBRARIES       += libadolcampi.la
libadolcampi_la_SOURCES  = ampisupport.cpp ampisupportAdolc.cpp \
                           ampisupportAdolc.h
endif

if BUILD_ADOLC_MEDIPACK_SUPPORT
noinst_LTLIBRARIES       += libadolcmedipack.la
libadolcmedipack_la_SOURCES  = medipacksupport.cpp \
                               medipacksupport_p.h
endif


libadolcsrc_la_SOURCES  = adalloc.c rpl_malloc.c adouble.cpp \
                       convolut.c fortutils.c \
                       interfaces.cpp interfacesf.c \
                       taping.c tape_handling.cpp \
                       dvlparms.h oplate.h taping_p.h rpl_malloc.h storemanager.h \
                       externfcts_p.h checkpointing_p.h buffer_temp.h \
                       zos_forward.c fos_forward.c fov_forward.c \
                       hos_forward.c hov_forward.c hov_wk_forward.c \
                       fos_reverse.c fov_reverse.c \
                       hos_reverse.c hos_ov_reverse.c hov_reverse.c \
                       forward_partx.c zos_pl_forward.c fos_pl_reverse.c fos_pl_sig_reverse.c \
                       fos_pl_forward.c fov_pl_forward.c fos_pl_sig_forward.c \
                       fov_pl_sig_forward.c externfcts.cpp checkpointing.cpp \
                       fixpoint.cpp fov_offset_forward.c revolve.c \
                       advector.cpp adouble_tl.cpp adouble_tl_indo.cpp adouble_tl_hov.cpp param.cpp externfcts2.cpp

if SPARSE
libadolcsrc_la_SOURCES  += int_forward_s.c int_forward_t.c \
                       indopro_forward_s.c indopro_forward_t.c \
                       indopro_forward_pl.c \
                       nonl_ind_forward_s.c nonl_ind_forward_t.c \
                       nonl_ind_old_forward_s.c nonl_ind_old_forward_t.c \
                       int_reverse_s.c int_reverse_t.c

endif
