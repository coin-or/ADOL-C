/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/psdrivers.h
 Revision: $Id$
 Contents: Easy to use drivers for piecewise smooth functions
           (with C and C++ callable interfaces including Fortran 
           callable versions).

 Copyright (c) Andrea Walther, Sabrina Fiege

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/
#if !defined(ADOLC_DRIVERS_PSDRIVERS_H)
#define ADOLC_DRIVERS_PSDRIVERS_H 1

#include <adolc/internal/common.h>
#include <adolc/interfaces.h>


BEGIN_C_DECLS

/****************************************************************************/
/*                                                 DRIVERS FOR PS FUNCTIONS */

/*--------------------------------------------------------------------------*/
/*                                              directional_active_gradient */
/*                                                                          */
ADOLC_DLL_EXPORT int directional_active_gradient(short,int,double*,short*,double*,double*,short*);  
ADOLC_DLL_EXPORT fint directional_active_gradient_(fint,fint,double*,short*,double*,double*,short*);  
/*--------------------------------------------------------------------------*/
/*                                                               abs_normal */
/*                                                                          */
ADOLC_DLL_EXPORT int abs_normal(short,int,int,int,double*,short*,double*,double*,double*,double*,double**,double**,double**,double**);
ADOLC_DLL_EXPORT fint abs_normal_(fint*,fint*,fint*,fint*,fdouble*,fint*,fdouble*,fdouble*,fdouble*,fdouble*,fdouble*,fdouble*,fdouble*,fdouble*);


END_C_DECLS

/****************************************************************************/
/****************************************************************************/
/*                                                       Now the C++ THINGS */
#if defined(__cplusplus)

/****************************************************************************/
/*                                                 DRIVERS FOR PS FUNCTIONS */

/*--------------------------------------------------------------------------*/
/*                                              directional_active_gradient */
/*                                                                          */
int directional_active_gradient(short tag,      /* trace identifier */
				int n,          /* number of independents */
				double* x,      /* value of independents */
				short *sigma_x, /* sigma of x */
				double* d,      /* direction */
				double* g,      /* directional active gradient */
				short *sigma_g  /* sigma of g */
				);

/*--------------------------------------------------------------------------*/
/*                                                               abs_normal */
/*                                                                          */
int abs_normal(short tag,      /* tape identifier */ 
               int m,          /* number of dependents   */             
               int n,          /* number of independents */
               double *x,      /* base point */ 
               short *sigma_x, /* sigma of x */
               double *y,      /* function value */
               double *z,      /* switching variables */
               double *cz,     /* first constant */
               double *cy,     /* second constant */
               double **a,      
               double **b,
               double **Z, 
               double **L);


#endif

/****************************************************************************/

#endif

