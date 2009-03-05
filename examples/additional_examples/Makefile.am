##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id: Makefile.am 299 2009-02-26 09:20:11Z awalther $
##
## Copyright (C) Andrea Walther, Andreas Kowarz
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

MAINTAINERCLEANFILES = Makefile.in *~

EXTRA_DIST           = Makefile.win

if ADDEXA
SUBDIRS              = clock hessmat lufact ode sparse tapesave timing \
                       detexam helm lighthouse scal speelpenning taylor pow \
                       checkpointing ext_diff_func fixpoint_exam openmp_exam
endif

if PAREXA
SUBDIRS              = clock hessmat lufact ode sparse tapesave timing \
                       detexam helm lighthouse scal speelpenning taylor pow \
                       checkpointing ext_diff_func fixpoint_exam openmp_exam
endif
