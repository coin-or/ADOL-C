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

#if !defined(ADOLC_FIXPOINT_H)
#define ADOLC_FIXPOINT_H 1

#include <adolc/internal/common.h>

BEGIN_C_DECLS

int fp_iteration ( int        sub_tape_num,
                   int      (*double_F)(double*, double* ,double*, int, int),
                   int      (*adouble_F)(adouble*, adouble*, adouble*, int, int),
                   double   (*norm)(double*, int),
                   double   (*norm_deriv)(double*, int),
                   double     epsilon,
                   double     epsilon_deriv,
                   int        N_max,
                   int        N_max_deriv,
                   adouble   *x_0,
                   adouble   *u,
                   adouble   *x_fix,
                   int        dim_x,
                   int        dim_u );

END_C_DECLS

#endif /* ADOLC_FIXPOINT_H */
