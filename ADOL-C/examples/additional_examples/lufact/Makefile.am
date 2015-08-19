##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id$
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
AM_CPPFLAGS          = -I$(top_builddir)/ADOL-C/include -I$(top_srcdir)/ADOL-C/include

LIBS                = -l${adolclib}
AM_LDFLAGS           = -L$(top_builddir)/ADOL-C

EXTRA_DIST           = LU.h

MAINTAINERCLEANFILES = Makefile.in tape.tex *~ *.orig

if ADDEXA
noinst_PROGRAMS      = LUsolve LUdet
endif

LUsolve_SOURCES      = LUsolve.cpp LU.cpp

LUdet_SOURCES        = LUdet.cpp LU.cpp
