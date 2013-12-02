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
SUBDIRS              = drivers sparse tapedoc
else
SUBDIRS              = drivers tapedoc
endif

MAINTAINERCLEANFILES = Makefile.in *~ *.orig adolc_config.h config.h.in

EXTRA_DIST           = uni5_for.c fo_rev.c ho_rev.c



if BUILD_ADOLC_AMPI_SUPPORT
lib_LTLIBRARIES      = libadolc_ampi.la
noinst_LTLIBRARIES   = libadolc.la
libadolc_ampi_la_LDFLAGS  = -version-info 2:1:1 -no-undefined
libadolc_la_LDFLAGS  =
libadolc_ampi_la_LIBADD = libadolc.la
libadolc_ampi_la_SOURCES = ampisupport.cpp ampisupportAdolc.cpp \
                           ampisupportAdolc.h
else
lib_LTLIBRARIES      = libadolc.la
libadolc_la_LDFLAGS  = -version-info 2:1:1 -no-undefined
endif


libadolc_la_SOURCES  = adalloc.c malloc.c adouble.cpp \
                       convolut.c fortutils.c \
                       interfaces.cpp interfacesf.c \
                       taping.c tape_handling.cpp \
                       oplate.h taping_p.h malloc.h storemanager.h \
                       externfcts_p.h checkpointing_p.h buffer_temp.h \
                       zos_forward.c fos_forward.c fov_forward.c \
                       hos_forward.c hov_forward.c hov_wk_forward.c \
                       fos_reverse.c fov_reverse.c \
                       hos_reverse.c hos_ov_reverse.c hov_reverse.c \
                       forward_partx.c fos_an_reverse.c \
                       externfcts.cpp checkpointing.cpp \
                       fixpoint.cpp fov_offset_forward.c revolve.c \
                       advector.cpp adouble_tl.cpp

libadolc_la_LIBADD   = drivers/libdrivers.la tapedoc/libtapedoc.la

if SPARSE
libadolc_la_SOURCES  += int_forward_s.c int_forward_t.c \
                       indopro_forward_s.c indopro_forward_t.c \
                       nonl_ind_forward_s.c nonl_ind_forward_t.c \
                       nonl_ind_old_forward_s.c nonl_ind_old_forward_t.c \
                       int_reverse_s.c int_reverse_t.c

libadolc_la_LIBADD   += sparse/libsparse.la
libadolc_la_LDFLAGS  += @COLPACK_LIBS@
endif
