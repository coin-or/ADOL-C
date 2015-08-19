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

AM_CFLAGS            = @ac_adolc_cflags@
AM_CXXFLAGS          = @ac_adolc_cxxflags@
AM_CPPFLAGS          = -I$(top_builddir)/ADOL-C/include -I$(top_srcdir)/ADOL-C/include

LIBS                = $(builddir)/../clock/libclock.la -l${adolclib}
AM_LDFLAGS           = -L$(top_builddir)/ADOL-C

MAINTAINERCLEANFILES = Makefile.in *~ *.orig

if ADDEXA
noinst_PROGRAMS      = speelpenning
endif

speelpenning_SOURCES = speelpenning.cpp
