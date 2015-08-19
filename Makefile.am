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

ACLOCAL_AMFLAGS = -I autoconf

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
                       MSVisualStudio/v14/adolc.sln MSVisualStudio/v14/adolc.vcxproj \
                       MSVisualStudio/v14/ColPack_Readme_VC++.txt \
                       MSVisualStudio/v14/ColPack.vcxproj \
                       MSVisualStudio/v14/sparse/config.h.in MSVisualStudio/v14/nosparse/config.h.in \
                       MSVisualStudio/v14/x64/sparse/config.h.in MSVisualStudio/v14/x64/nosparse/config.h.in \
                       MSVisualStudio/v14/sparse/config.h MSVisualStudio/v14/nosparse/config.h \
                       MSVisualStudio/v14/x64/sparse/config.h MSVisualStudio/v14/x64/nosparse/config.h \
                       MSVisualStudio/v14/Readme_VC++.txt \
                       MSVisualStudio/v14/installer.sh \
                       adolc.spec.in adolc.spec \
                       update_versions.sh

SUBDIRS              = ADOL-C

# install pkg-config information
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = @adolclib@.pc

@adolclib@.pc: adolclib.pc
	mv $< $@

test: all
	cd ADOL-C; $(MAKE) test

distclean-local:
	rm -f @adolclib@.pc

.PHONY: test
