##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id: Makefile.am 300 2009-02-26 10:42:44Z awalther $
##
## Copyright (C) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
##               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
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

INCLUDES             = -I$(top_builddir) -I$(top_builddir)/examples/additional_examples

EXTRA_DIST           = LU.h Makefile.win

MAINTAINERCLEANFILES = Makefile.in tape.tex *~ *.orig

if ADDEXA
noinst_PROGRAMS      = LUsolve LUdet

LUsolve_SOURCES      = LUsolve.cpp LU.cpp

LUdet_SOURCES        = LUdet.cpp LU.cpp
endif
