##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id$
##
## Copyright (C) Kshitij Kulshreshtha
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

AM_CFLAGS            = @ac_adolc_cflags@ -DADOLC_INTERNAL=1 -std=gnu99
AM_CXXFLAGS          = @ac_adolc_cxxflags@ -DADOLC_INTERNAL=1

AM_CPPFLAGS          = -I$(top_builddir)/ADOL-C/include -I$(top_srcdir)/ADOL-C/include -I$(top_srcdir)/ADOL-C/src -I$(top_srcdir)/ADOL-C/src/ampi
AM_LDFLAGS           = -L$(top_builddir)/ADOL-C/src

lib_LTLIBRARIES      = libadolc_ampi.la

CC = ${MPICC}
CXX = ${MPICXX}

am__v_CC_0 = @echo "  MPICC   " $@;
am__v_CCLD_0 = @echo "  MPICCLD " $@;
am__v_CXX_0 = @echo "  MPICXX  " $@;
am__v_CXXLD_0 = @echo "  MPICXXLD" $@;


libadolc_ampi_la_CPPFLAGS = ${AMPICPPFLAGS} ${AM_CPPFLAGS}
libadolc_ampi_la_SOURCES = ampisupport.cpp ampisupportAdolc.h ampisupportAdolc.cpp ampiTraverseStub.c
libadolc_ampi_la_LIBADD = $(top_builddir)/ADOL-C/src/libadolc.la
libadolc_ampi_la_LDFLAGS = ${AMPILDFLAGS} ${AMPILIBS} ${AM_LDFLAGS}

