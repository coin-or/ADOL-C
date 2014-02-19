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

AM_CFLAGS               = @ac_adolc_cflags@
AM_CXXFLAGS             = @ac_adolc_cxxflags@
AM_CPPFLAGS             = -I$(top_builddir)/ADOL-C/include -I$(top_srcdir)/ADOL-C/include

LDADD                = -l${adolclib}
AM_LDFLAGS           = -L$(top_builddir)/ADOL-C

MAINTAINERCLEANFILES = Makefile.in *~ *.orig

if ADDEXA
noinst_PROGRAMS      = odexam od2exam od3exam
endif

odexam_SOURCES       = vfunc_robertson.cpp vgenodemain.cpp $(srcdir)/../clock/myclock.cpp

od2exam_SOURCES      = vfunc_eutroph.cpp vgenodemain.cpp $(srcdir)/../clock/myclock.cpp

od3exam_SOURCES      = vfunc_fic.cpp vgenodemain.cpp $(srcdir)/../clock/myclock.cpp
