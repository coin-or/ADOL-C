##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id: Makefile.am 61 2009-12-07 14:49:34Z awalther $
##
## Copyright (C) Andrea Walther
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

noinst_PROGRAMS         = powexam speelpenning fminmax

speelpenning_SOURCES    = speelpenning.cpp

powexam_SOURCES         = powexam.cpp

fminmax_SOURCES		= fminmax.cpp

powexam_LDADD           = ../lib${adolclib}.la
speelpenning_LDADD = ../lib${adolclib}.la
fminmax_LDADD		= ../lib${adolclib}.la

AM_CFLAGS               = @ac_adolc_cflags@
AM_CXXFLAGS             = @ac_adolc_cxxflags@

# Here list all include flags, relative to this "srcdir" directory.  This
# "cygpath" stuff is necessary to compile with native compilers on Cygwin
AM_CPPFLAGS = -I$(top_builddir)/ADOL-C/include  -I$(top_srcdir)/ADOL-C/include

EXTRA_DIST = run_tests

# This line is necessary to allow VPATH compilation with MS compilers
# on Cygwin
DEFAULT_INCLUDES = -I. -I$(srcdir) -I$(top_builddir)/inc

test:	powexam speelpenning 
	chmod u+x $(srcdir)/run_tests
	$(srcdir)/run_tests

