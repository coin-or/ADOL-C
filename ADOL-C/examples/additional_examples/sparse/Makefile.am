##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id: Makefile.am 37 2009-05-28 12:56:44Z awalther $
##
## Copyright (C) Andrea Walther, Andreas Kowarz
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

AM_CFLAGS               = @ac_adolc_cflags@
AM_CXXFLAGS             = @ac_adolc_cxxflags@
AM_CPPFLAGS             = -I$(top_srcdir)/ADOL-C/src

if SPARSE
LDADD                   = -ladolc $(libdir)/libColPack.a
AM_LDFLAGS              = -L../../../src
else
LDADD                   = -ladolc 
AM_LDFLAGS              = -L../../../src
endif

INCLUDES                = -I$(top_builddir) -I$(top_builddir)/examples/additional_examples

MAINTAINERCLEANFILES    = Makefile.in *~ *.orig

EXTRA_DIST              = Makefile.win

if ADDEXA
if SPARSE
noinst_PROGRAMS         = sparse_jacobian sparse_hessian jacpatexam

sparse_jacobian_SOURCES = sparse_jacobian.cpp

sparse_hessian_SOURCES  = sparse_hessian.cpp

jacpatexam_SOURCES      = jacpatexam.cpp ../clock/myclock.cpp
endif
endif
