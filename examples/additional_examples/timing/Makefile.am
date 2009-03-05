##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id: Makefile.am 300 2009-02-26 10:42:44Z awalther $
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
AM_CPPFLAGS          = -I$(top_srcdir) 

if SPARSE
LDADD                   = -ladolc $(libdir)/libColPack.a
AM_LDFLAGS              = -L../../../adolc
else
LDADD                   = -ladolc 
AM_LDFLAGS              = -L../../../adolc
endif

INCLUDES                = -I$(top_builddir) -I$(top_builddir)/examples/additional_examples

EXTRA_DIST              = rotations.h *.ctrl Makefile.win

MAINTAINERCLEANFILES    = Makefile.in in_tape.1 op_tape.1 vs_tape1 *~ *.orig

if ADDEXA
noinst_PROGRAMS         = detexam speelpenning griewankexam michalewitzexam \
                          rosenbrockexam powexam helmholtzexam shuttlexam \
                          gearexam pargearexam simplevec eutrophexam \
                          robertsonexam ficexam experimental

detexam_SOURCES         = sfunc_determinant.cpp sgenmain.cpp \
                          ../clock/myclock.cpp

speelpenning_SOURCES    = sfunc_speelpenning.cpp sgenmain.cpp \
                          ../clock/myclock.cpp

griewankexam_SOURCES    = sfunc_griewank.cpp sgenmain.cpp \
                          ../clock/myclock.cpp

michalewitzexam_SOURCES = sfunc_michalewitz.cpp sgenmain.cpp \
                          ../clock/myclock.cpp

rosenbrockexam_SOURCES  = sfunc_rosenbrock.cpp sgenmain.cpp \
                          ../clock/myclock.cpp

powexam_SOURCES         = sfunc_power.cpp sgenmain.cpp \
                          ../clock/myclock.cpp

helmholtzexam_SOURCES   = sfunc_helmholtz.cpp sgenmain.cpp \
                          ../clock/myclock.cpp

shuttlexam_SOURCES      = vfunc_shuttle.cpp vgenmain.cpp \
                          ../clock/myclock.cpp

gearexam_SOURCES        = vfunc_gear.cpp vgenmain.cpp \
                          ../clock/myclock.cpp

pargearexam_SOURCES     = vfunc_pargear.cpp rotations.cpp vgenmain.cpp \
                          ../clock/myclock.cpp

simplevec_SOURCES       = vfunc_simplevec.cpp vgenmain.cpp \
                          ../clock/myclock.cpp

eutrophexam_SOURCES     = ../ode/vfunc_eutroph.cpp vgenmain.cpp \
                          ../clock/myclock.cpp

robertsonexam_SOURCES   = ../ode/vfunc_robertson.cpp vgenmain.cpp \
                          ../clock/myclock.cpp

ficexam_SOURCES         = ../ode/vfunc_fic.cpp vgenmain.cpp \
                          ../clock/myclock.cpp

experimental_SOURCES    = sfunc_experimental.cpp sgenmain.cpp \
                          ../clock/myclock.cpp
endif
