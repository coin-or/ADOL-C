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

AM_CFLAGS               = @ac_adolc_cflags@
AM_CXXFLAGS             = @ac_adolc_cxxflags@
AM_CPPFLAGS             = -I$(top_builddir)/ADOL-C/include -I$(top_srcdir)/ADOL-C/include

LIBS                   = -l${adolclib}
AM_LDFLAGS              = -L$(top_builddir)/ADOL-C

MAINTAINERCLEANFILES    = Makefile.in *~ *.orig ADOLC-*.tap

EXTRA_DIST              = adolc_examples.sln \
                          detexam.vcxproj speelpenning.vcxproj odexam.vcxproj \
                          tapeless_scalar.vcxproj tapeless_vector.vcxproj \
                          Readme_VC++.txt

if ADDEXA
SUBDIRS                 = additional_examples
endif

if PAREXA
SUBDIRS                 = additional_examples
endif

if DOCEXA
noinst_PROGRAMS         = detexam speelpenning powexam odexam luexam \
                          tapeless_scalar tapeless_vector traceless_higher_order
if SPARSE
noinst_PROGRAMS        += traceless_vector_indo
endif
endif

detexam_SOURCES         = detexam.cpp

speelpenning_SOURCES    = speelpenning.cpp

powexam_SOURCES         = powexam.cpp

odexam_SOURCES          = odexam.cpp 

luexam_SOURCES          = luexam.cpp

tapeless_scalar_SOURCES = tapeless_scalar.cpp

tapeless_vector_SOURCES = tapeless_vector.cpp

if SPARSE
traceless_vector_indo_SOURCES = traceless_vector_indo.cpp
endif

traceless_higher_order_SOURCES= traceless_higher_order.cpp
