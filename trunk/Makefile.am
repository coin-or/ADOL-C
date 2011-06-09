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

EXTRA_DIST           = TODO BUGS LICENSE update_versions.sh \
                       ADOL-C/doc/version.tex.in ADOL-C/doc/version.tex \
                       ADOL-C/doc/adolc-manual.pdf ADOL-C/doc/adolc-manual.ps \
                       ADOL-C/doc/adolc-manual.tex ADOL-C/doc/short_ref.pdf \
                       ADOL-C/doc/short_ref.ps ADOL-C/doc/short_ref.tex \
                       ADOL-C/doc/multiplexed.pdf ADOL-C/doc/multiplexed.eps \
                       ADOL-C/doc/tapeadv.pdf ADOL-C/doc/tapeadv.eps \
                       ADOL-C/doc/tapebasic.pdf ADOL-C/doc/tapebasic.eps \
                       ADOL-C/doc/tap_point.pdf ADOL-C/doc/tap_point.eps \
                       autoconf/shave.in autoconf/shave-libtool.in \
                       autoconf/dolt.m4 autoconf/shave.m4 \
                       MSVisualStudio/v10/adolc.sln MSVisualStudio/v10/adolc.vcxproj \
                       MSVisualStudio/v10/sparse/config.h.in MSVisualStudio/v10/nosparse/config.h.in \
                       MSVisualStudio/v10/x64/sparse/config.h.in MSVisualStudio/v10/x64/nosparse/config.h.in \
                       MSVisualStudio/v10/sparse/config.h MSVisualStudio/v10/nosparse/config.h \
                       MSVisualStudio/v10/x64/sparse/config.h MSVisualStudio/v10/x64/nosparse/config.h \
                       MSVisualStudio/v10/Readme_VC++.txt \
                       MSVisualStudio/v10/installer.sh \
                       adolc.spec.in adolc.spec \
                       update_versions.sh \
                       ThirdParty/ColPack/ColPack.vcxproj \
                       ThirdParty/ColPack/Readme_VC++.txt

SUBDIRS              = ADOL-C

MAINTAINERCLEANFILES = Makefile.in aclocal.m4 configure ADOL-C/src/config.h.in \
                       ADOL-C/src/stamp-h.in *~ *.orig 

DISTCLEANFILES = doltcompile doltlibtool

test: all
	cd ADOL-C; $(MAKE) test

