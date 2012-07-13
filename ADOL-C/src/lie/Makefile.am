##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id$
##
## Copyright (C)  Andrea Walther, Andreas Kowarz, Kshitij Kulshreshtha
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

AM_CFLAGS                = @ac_adolc_cflags@ -DADOLC_INTERNAL=1 
AM_CXXFLAGS              = @ac_adolc_cxxflags@ -DADOLC_INTERNAL=1 
AM_CPPFLAGS               = -I$(top_srcdir)/ADOL-C -I$(top_srcdir)/ADOL-C/src  

INCLUDES                 = -I$(top_builddir)/ADOL-C

MAINTAINERCLEANFILES     = Makefile.in *~ *.orig

noinst_LTLIBRARIES       = liblie.la

liblieincludedir      = $(pkgincludedir)/lie

liblieinclude_HEADERS = adolc_lie.h 

liblie_la_SOURCES     = adolc_lie_c.c adolc_lie.cpp

