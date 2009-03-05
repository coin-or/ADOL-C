##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id: Makefile.am 267 2008-12-15 13:19:25Z awalther $
##
## Copyright (C) Andrea Walther
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

AM_CFLAGS            = @ac_adolc_cflags@
AM_CXXFLAGS          = @ac_adolc_cxxflags@
AM_CPPFLAGS          = -I$(top_srcdir) 

if SPARSE
LDADD                = -ladolc $(libdir)/libColPack.a
AM_LDFLAGS           = -L../../../adolc
else
LDADD                = -ladolc 
AM_LDFLAGS           = -L../../../adolc
endif

INCLUDES             = -I$(top_builddir)

MAINTAINERCLEANFILES = Makefile.in *~ *.orig

EXTRA_DIST           = Makefile.win

if PAREXA
noinst_PROGRAMS      = liborser liborpar

liborser_SOURCES     = liborser.cpp

liborpar_SOURCES     = liborpar.cpp

endif
