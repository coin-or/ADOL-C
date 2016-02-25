##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id$
##
## Copyright (C)  Andrea Walther, Andreas Kowarz
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

AM_CFLAGS                = @ac_adolc_cflags@ -DADOLC_INTERNAL=1 -DSPARSE=1
AM_CXXFLAGS              = @ac_adolc_cxxflags@ -DADOLC_INTERNAL=1 -DSPARSE=1
AM_CPPFLAGS               = -I$(top_builddir)/ADOL-C/include -I$(top_srcdir)/ADOL-C/include -I$(top_srcdir)/ADOL-C/src

MAINTAINERCLEANFILES     = Makefile.in *~ *.orig

noinst_LTLIBRARIES       = libsparse.la

libsparse_la_SOURCES     = sparse_fo_rev.cpp sparsedrivers.cpp
