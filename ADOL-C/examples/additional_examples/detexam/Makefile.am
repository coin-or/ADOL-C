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
AM_CPPFLAGS          = -I$(top_srcdir)/ADOL-C

LDADD                = -ladolc 
AM_LDFLAGS           = -L$(top_builddir)/ADOL-C/src

INCLUDES             = -I$(top_builddir)/ADOL-C -I$(top_srcdir)/ADOL-C/examples/additional_examples

MAINTAINERCLEANFILES = Makefile.in *~ *.orig

if ADDEXA
noinst_PROGRAMS      = detexam detexam-old
endif

detexam_SOURCES      = detexam.cpp ../clock/myclock.cpp

detexam_old_SOURCES  = detexam-old.cpp ../clock/myclock.cpp
