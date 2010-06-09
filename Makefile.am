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

AUTOMAKE_OPTIONS     = foreign

EXTRA_DIST           = TODO BUGS Makefile.win make_32 make_64 \
                       comp_win.bat winflags_dll winflags_exa

SUBDIRS              = ADOL-C

MAINTAINERCLEANFILES = Makefile.in aclocal.m4 configure ADOL-C/src/config.h.in \
                       ADOL-C/src/stamp-h.in *~ *.orig 

DISTCLEANFILES = doltcompile doltlibtool

test: all
	cd ADOL-C; $(MAKE) test

