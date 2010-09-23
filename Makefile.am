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

EXTRA_DIST           = TODO BUGS update_versions.sh \
                       ADOL-C/doc/version.tex.in ADOL-C/doc/version.tex \
                       autoconf/shave.in autoconf/shave-libtool.in \
                       autoconf/dolt.m4 autoconf/shave.m4 \
                       windows/adolc.sln windows/adolc.vcxproj \
                       windows/sparse/config.h.in windows/nosparse/config.h.in \
                       windows/x64/sparse/config.h.in windows/x64/nosparse/config.h.in \
                       windows/sparse/config.h windows/nosparse/config.h \
                       windows/x64/sparse/config.h windows/x64/nosparse/config.h \
                       windows/Readme_VC++.txt \
                       windows/vcredist_x86.exe windows/x64/vcredist_x64.exe \
                       windows/installer.sh \
                       adolc.spec.in adolc.spec \
                       ThirdParty/ColPack/ColPack.vcxproj \
                       ThirdParty/ColPack/colpack_vcxcompile.patch \
                       ThirdParty/ColPack/Readme_VC++.txt

SUBDIRS              = ADOL-C

MAINTAINERCLEANFILES = Makefile.in aclocal.m4 configure ADOL-C/src/config.h.in \
                       ADOL-C/src/stamp-h.in *~ *.orig 

DISTCLEANFILES = doltcompile doltlibtool

test: all
	cd ADOL-C; $(MAKE) test

