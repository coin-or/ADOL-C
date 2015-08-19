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

LIBS                   = $(builddir)/../clock/libclock.la -l${adolclib}
AM_LDFLAGS              = -L$(top_builddir)/ADOL-C

CTRLFILES               = detexam.ctrl \
                          eutrophexam.ctrl \
                          experimental.ctrl \
                          ficexam.ctrl \
                          gearexam.ctrl \
                          griewankexam.ctrl \
                          helmholtzexam.ctrl \
                          michalewitzexam.ctrl \
                          pargearexam.ctrl \
                          powexam.ctrl \
                          robertsonexam.ctrl \
                          rosenbrockexam.ctrl \
                          shuttlexam.ctrl \
                          simplevec.ctrl \
                          speelpenning.ctrl


EXTRA_DIST              = rotations.h $(CTRLFILES)

MAINTAINERCLEANFILES    = Makefile.in in_tape.1 op_tape.1 vs_tape1 *~ *.orig

if ADDEXA
noinst_PROGRAMS         = detexam speelpenning griewankexam michalewitzexam \
                          rosenbrockexam powexam helmholtzexam shuttlexam \
                          gearexam pargearexam simplevec eutrophexam \
                          robertsonexam ficexam experimental
endif

detexam_SOURCES         = sfunc_determinant.cpp sgenmain.cpp

speelpenning_SOURCES    = sfunc_speelpenning.cpp sgenmain.cpp

griewankexam_SOURCES    = sfunc_griewank.cpp sgenmain.cpp

michalewitzexam_SOURCES = sfunc_michalewitz.cpp sgenmain.cpp

rosenbrockexam_SOURCES  = sfunc_rosenbrock.cpp sgenmain.cpp

powexam_SOURCES         = sfunc_power.cpp sgenmain.cpp

helmholtzexam_SOURCES   = sfunc_helmholtz.cpp sgenmain.cpp

shuttlexam_SOURCES      = vfunc_shuttle.cpp vgenmain.cpp

gearexam_SOURCES        = vfunc_gear.cpp vgenmain.cpp

pargearexam_SOURCES     = vfunc_pargear.cpp rotations.cpp vgenmain.cpp

simplevec_SOURCES       = vfunc_simplevec.cpp vgenmain.cpp

eutrophexam_SOURCES     = vgenmain.cpp

eutrophexam_LDADD       = $(builddir)/../ode/libeutroph.la

robertsonexam_SOURCES   = vgenmain.cpp

robertsonexam_LDADD     = $(builddir)/../ode/librobertson.la

ficexam_SOURCES         = vgenmain.cpp

ficexam_LDADD           = $(builddir)/../ode/libfic.la

experimental_SOURCES    = sfunc_experimental.cpp sgenmain.cpp
