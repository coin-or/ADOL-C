##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id: Makefile.am 38 2009-06-17 15:48:09Z awalther $
##
## Copyright (C) Andrea Walther, Andreas Kowarz, Kshitij Kulshreshtha
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

AUTOMAKE_OPTIONS     = foreign

if DOCEXA
SUBDIRS              = src examples
else
if ADDEXA
SUBDIRS              = src examples
else
if PAREXA
SUBDIRS              = src examples 
else
SUBDIRS              = src
endif
endif
endif

SUBDIRS += include

if BUILD_ADOLC_AMPI_SUPPORT 
SUBDIRS += ampi
endif

SUBDIRS += test

MAINTAINERCLEANFILES = Makefile.in 

test: all
	cd test; $(MAKE) test


