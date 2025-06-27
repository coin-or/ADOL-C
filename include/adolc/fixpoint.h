/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fixpoint.h
 Revision: $Id$
 Contents: all C functions directly accessing at least one of the four tapes
           (operations, locations, constants, value stack)

 Copyright (c) Andreas Kowarz, Sebastian Schlenkrich

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#ifndef ADOLC_FIXPOINT_H
#define ADOLC_FIXPOINT_H

#include <adolc/internal/common.h>

typedef int (*double_F)(double *, double *, double *, int, int);
typedef int (*adouble_F)(adouble *, adouble *, adouble *, int, int);
typedef double (*norm_F)(double *, int);
typedef double (*norm_deriv_F)(double *, int);

int fp_iteration(short tapeId, size_t sub_tape_num, double_F, adouble_F, norm_F,
                 norm_deriv_F, double epsilon, double epsilon_deriv,
                 size_t N_max, size_t N_max_deriv, adouble *x_0, adouble *u,
                 adouble *x_fix, size_t dim_x, size_t dim_u);

#endif // ADOLC_FIXPOINT_H
