##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id$
##
## Copyright (C) Kshitij Kulshreshtha
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

ACLOCAL_AMFLAGS = -I autoconf

EXTRA_DIST = swigprocess.py adolc-r.i adolc-python.i adubswigfuncs.hpp adolc_all_in.hpp matrixmemory.hpp pyedfclasses.hpp

clean-local:
	-rm -rf python R *.cpp *.cxx *.o *.h include

if PYTHONFOUND
if SPARSE
install:
	CXX=${CXX} ${PYTHON} ${srcdir}/setup.py build --only-swig --lib-prefix=${prefix} --lib-libdir=${libdir} @python_srcbase@ install --prefix=${DESTDIR}/${prefix}
endif
endif
