/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse/sparsedrivers.h
 Revision: $Id$
 Contents: This file contains some "Easy To Use" interfaces of the sparse
package.

 Copyright (c) Andrea Walther

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#ifndef ADOLC_SPARSE_DRIVERS_H
#define ADOLC_SPARSE_DRIVERS_H

#include <adolc/internal/common.h>

namespace ADOLC::Sparse {

BEGIN_C_DECLS

/****************************************************************************/

/*--------------------------------------------------------------------------*/
/*                                                         jacobian pattern */
/* jac_pat(tag, m, n, argument,                                             */
/*         crs[] [ crs[][0] = non-zero independent blocks per row ],        */
/*         options[3])                                                      */
/*                                                                          */

ADOLC_API int jac_pat(short, int, int, const double *, unsigned int **, int *);

/*--------------------------------------------------------------------------*/
/*                                              abs-normal jacobian pattern */
/* absnormal_jac_pat(tag, m, n, s, argument,                                */
/*         crs[] [ crs[][0] = non-zero independent blocks per row ])        */
/*                                                                          */
ADOLC_API int absnormal_jac_pat(short, int, int, int, const double *,
                                unsigned int **);
/*--------------------------------------------------------------------------*/
/*                                         seed matrix for sparse jacobian  */
/* generate_seed_jac(m, n, crs, &seed, &p, option);                         */

ADOLC_API void generate_seed_jac(int, int, unsigned int **, double ***, int *,
                                 int);

/*--------------------------------------------------------------------------*/
/*                                                         sparse jacobian  */
/* int sparse_jac(tag, m, n, repeat, x, &nnz, &row_ind, &col_ind, &values,  */
/*                options[3]);                                              */

ADOLC_API int sparse_jac(short, int, int, int, const double *, int *,
                         unsigned int **, unsigned int **, double **, int *);

/*--------------------------------------------------------------------------*/
/*                                                          hessian pattern */
/* hess_pat(tag, n, x[n], crs[n][*], option)                                */
/*                                                                          */
/*     crs[i][ crs[i][0] = non-zero entries per row ]                       */
/*                                                                          */

ADOLC_API int hess_pat(short, int, const double *, unsigned int **, int);

/*--------------------------------------------------------------------------*/
/*                                          seed matrix for sparse hessian  */
/* generate_seed_hess(n, crs, &seed, &p, option);                           */

ADOLC_API void generate_seed_hess(int, unsigned int **, double ***, int *, int);

/*--------------------------------------------------------------------------*/
/*                                                          sparse hessian  */
/* int sparse_hess(tag, n, repeat, x, &nnz, &row_ind, &col_ind, &values,    */
/*                 options[2]);                                             */

ADOLC_API int sparse_hess(short, int, int, const double *, int *,
                          unsigned int **, unsigned int **, double **, int *);

ADOLC_API void set_HP(short tag, /* tape identification */
                      int indep, /* number of independent variables */
                      unsigned int **HP);

ADOLC_API void get_HP(short tag, /* tape identification */
                      int indep, /* number of independent variables */
                      unsigned int ***HP);

/*--------------------------------------------------------------------------*/
/*                                                   JACOBIAN BLOCK PATTERN */

/* Max. number of unsigned ints to store the seed / jacobian matrix strips.
   Reduce this value to x if your system happens to run out of memory.
   x < 10 makes no sense. x = 50 or 100 is better
   x stays for ( x * sizeof(size_t) * 8 )
   (block) variables at once                                            */

#define PQ_STRIPMINE_MAX 30

ADOLC_API int bit_vector_propagation(short, int, int, const double *,
                                     unsigned int **, int *);

/****************************************************************************/

END_C_DECLS

} // namespace ADOLC::Sparse

#endif // ADOLC_SPARSE_DRIVERS_H
