/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/drivers.h
 Revision: $Id$
 Contents: Easy to use drivers for optimization and nonlinear equations
           (with C and C++ callable interfaces including Fortran 
            callable versions).
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/
#if !defined(ADOLC_DRIVERS_DRIVERS_H)
#define ADOLC_DRIVERS_DRIVERS_H 1

#include <adolc/internal/common.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                         DRIVERS FOR OPTIMIZATION AND NONLINEAR EQUATIONS */

/*--------------------------------------------------------------------------*/
/*                                                                 function */
/* function(tag, m, n, x[n], y[m])                                          */
ADOLC_DLL_EXPORT int function(short,int,int,double*,double*);
ADOLC_DLL_EXPORT fint function_(fint*,fint*,fint*,fdouble*,fdouble*);

/*--------------------------------------------------------------------------*/
/*                                                                 gradient */
/* gradient(tag, n, x[n], g[n])                                             */
ADOLC_DLL_EXPORT int gradient(short,int,const double*,double*);
ADOLC_DLL_EXPORT fint gradient_(fint*,fint*,fdouble*,fdouble*);

/*--------------------------------------------------------------------------*/
/*                                                                 jacobian */
/* jacobian(tag, m, n, x[n], J[m][n])                                       */
ADOLC_DLL_EXPORT int jacobian(short,int,int,const double*,double**);
ADOLC_DLL_EXPORT fint jacobian_(fint*,fint*,fint*,fdouble*,fdouble*);

/*--------------------------------------------------------------------------*/
/*                                                           large_jacobian */
/* large_jacobian(tag, m, n, k, x[n], y[m], J[m][n])                        */
ADOLC_DLL_EXPORT int large_jacobian(short,int,int,int,double*,double*,double**);
ADOLC_DLL_EXPORT fint large_jacobian_(fint*,fint*,fint*,fint*,fdouble*,fdouble*,fdouble*);


/*--------------------------------------------------------------------------*/
/*                                                         vector_jacobian  */
/* vec_jac(tag, m, n, repeat, x[n], u[m], v[n])                             */
ADOLC_DLL_EXPORT int vec_jac(short,int,int,int,double*,double*,double*);
ADOLC_DLL_EXPORT fint vec_jac_(fint*,fint*,fint*,fint*,
                               fdouble*,fdouble*,fdouble*);

/*--------------------------------------------------------------------------*/
/*                                                          jacobian_vector */
/* jac_vec(tag, m, n, x[n], v[n], u[m]);                                    */
ADOLC_DLL_EXPORT int jac_vec(short,int,int,double*,double*,double*);
ADOLC_DLL_EXPORT fint jac_vec_(fint*,fint*,fint*,fdouble*,fdouble*,fdouble*);

/*--------------------------------------------------------------------------*/
/*                                                                  hessian */
/* hessian(tag, n, x[n], lower triangle of H[n][n])                         */
/* uses Hessian-vector product                                              */
ADOLC_DLL_EXPORT int hessian(short,int,double*,double**);
ADOLC_DLL_EXPORT fint hessian_(fint*,fint*,fdouble*,fdouble*);

/*--------------------------------------------------------------------------*/
/*                                                                 hessian2 */
/* hessian2(tag, n, x[n], lower triangle of H[n][n])                        */
/* uses Hessian-matrix product                                              */
ADOLC_DLL_EXPORT int hessian2(short,int,double*,double**);
ADOLC_DLL_EXPORT fint hessian2_(fint*,fint*,fdouble*,fdouble*);

/*--------------------------------------------------------------------------*/
/*                                                           hessian_vector */
/* hess_vec(tag, n, x[n], v[n], w[n])                                       */
ADOLC_DLL_EXPORT int hess_vec(short,int,double*,double*,double*);
ADOLC_DLL_EXPORT fint hess_vec_(fint*,fint*,fdouble*,fdouble*,fdouble*);

/*--------------------------------------------------------------------------*/
/*                                                           hessian_matrix */
/* hess_mat(tag, n, q, x[n], V[n][q], W[n][q])                              */
ADOLC_DLL_EXPORT int hess_mat(short,int,int,double*,double**,double**);
ADOLC_DLL_EXPORT fint hess_mat_(fint*,fint*,fint*,
                                fdouble*,fdouble**,fdouble**);

/*--------------------------------------------------------------------------*/
/*                                                  lagrange_hessian_vector */
/* lagra_hess_vec(tag, m, n, x[n], v[n], u[m], w[n])                        */
ADOLC_DLL_EXPORT int lagra_hess_vec(short,int,int,double*,
                                    double*,double*,double*);
ADOLC_DLL_EXPORT fint lagra_hess_vec_(fint*,fint*,fint*,
                                      fdouble*,fdouble*,fdouble*,fdouble*);

END_C_DECLS

/****************************************************************************/
#endif

