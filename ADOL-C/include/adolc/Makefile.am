##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id$
##
## Copyright (C) Andrea Walther, Andreas Kowarz, Kshitij Kulshreshtha
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

pkginclude_HEADERS   = adolc.h adalloc.h adouble.h adutils.h adutilsc.h \
                       convolut.h fortutils.h \
                       interfaces.h taping.h \
                       externfcts.h checkpointing.h fixpoint.h\
                       adolc_sparse.h adolc_openmp.h \
                       revolve.h advector.h \
                       adtl.h adoublecuda.h

SUBDIRS = internal drivers tapedoc

if SPARSE
SUBDIRS += sparse
endif
