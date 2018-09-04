/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/pardrivers.h
 Revision: $Id$
 Contents: Easy to use OpenMP-parallel drivers for optimization and nonlinear
           equations (with C and C++ callable interfaces including Fortran
           callable versions).

 Copyright (c) Kshitij Kulshreshtha, Martin Schroschk

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_DRIVERS_PARDRIVERS_H)
#define ADOLC_DRIVERS_PARDRIVERS_H 1

#include <adolc/internal/common.h>

/****************************************************************************/

/*--------------------------------------------------------------------------*/
/*                                                             par_jacobian */
/* par_jacobian(tag, m, n, x[n], J[m][n])                                   */
ADOLC_DLL_EXPORT int par_jacobian(short, int, int, const double*, double**);
ADOLC_DLL_EXPORT int par_jac_mat(short, int, int, int, const double*,
                                 double**, double**);

ADOLC_DLL_EXPORT int par_mat_jac(short, int, int, int, const double*,
                                 double**, double**);

/****************************************************************************/
#endif
