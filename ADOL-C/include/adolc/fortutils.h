/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fortutils.h
 Revision: $Id$
 Contents: Internal tools to handle Fortran arrays

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_FORTUTILS_H)
#define ADOLC_FORTUTILS_H 1

#include <adolc/adolcexport.h>
#include <adolc/internal/common.h>

/****************************************************************************/
/*                                                         Now the C THINGS */
BEGIN_C_DECLS

ADOLC_API void spread1(int m, const fdouble *x, double *X);
ADOLC_API void pack1(int m, const double *X, fdouble *x);

ADOLC_API void spread2(int m, int n, const fdouble *x, double **X);
ADOLC_API void pack2(int m, int n, const double *const *X, fdouble *x);

ADOLC_API void spread3(int m, int n, int p, const fdouble *x, double ***X);
ADOLC_API void pack3(int m, int n, int p, const double *const *const *X,
                     fdouble *x);

END_C_DECLS

/****************************************************************************/
#endif
