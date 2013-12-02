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

LDADD                   = -l${adolclib}
AM_LDFLAGS              = -L$(top_builddir)/ADOL-C/src

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

detexam_SOURCES         = sfunc_determinant.cpp sgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

speelpenning_SOURCES    = sfunc_speelpenning.cpp sgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

griewankexam_SOURCES    = sfunc_griewank.cpp sgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

michalewitzexam_SOURCES = sfunc_michalewitz.cpp sgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

rosenbrockexam_SOURCES  = sfunc_rosenbrock.cpp sgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

powexam_SOURCES         = sfunc_power.cpp sgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

helmholtzexam_SOURCES   = sfunc_helmholtz.cpp sgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

shuttlexam_SOURCES      = vfunc_shuttle.cpp vgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

gearexam_SOURCES        = vfunc_gear.cpp vgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

pargearexam_SOURCES     = vfunc_pargear.cpp rotations.cpp vgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

simplevec_SOURCES       = vfunc_simplevec.cpp vgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

eutrophexam_SOURCES     = $(srcdir)/../ode/vfunc_eutroph.cpp vgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

robertsonexam_SOURCES   = $(srcdir)/../ode/vfunc_robertson.cpp vgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

ficexam_SOURCES         = $(srcdir)/../ode/vfunc_fic.cpp vgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp

experimental_SOURCES    = sfunc_experimental.cpp sgenmain.cpp \
                          $(srcdir)/../clock/myclock.cpp
