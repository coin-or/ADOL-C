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

#include <adolc/internal/common.h>

/****************************************************************************/
/*                                                         Now the C THINGS */
BEGIN_C_DECLS

ADOLC_DLL_EXPORT void spread1(int m, fdouble* x, double* X);
ADOLC_DLL_EXPORT void pack1(int m, double* X, fdouble* x);

ADOLC_DLL_EXPORT void spread2(int m, int n, fdouble* x, double** X);
ADOLC_DLL_EXPORT void pack2(int m, int n, double** X, fdouble* x);

ADOLC_DLL_EXPORT void spread3(int m, int n, int p, fdouble* x, double*** X);
ADOLC_DLL_EXPORT void pack3(int m, int n, int p, double*** X, fdouble* x);

END_C_DECLS

/****************************************************************************/
#endif
