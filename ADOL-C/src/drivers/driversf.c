/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/driversf.c
 Revision: $Id$
 Contents: Easy to use drivers for optimization and nonlinear equations
           (Implementation of the Fortran callable interfaces).
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
  
----------------------------------------------------------------------------*/
#include <adolc/drivers/drivers.h>
#include <adolc/interfaces.h>
#include <adolc/adalloc.h>
#include <adolc/fortutils.h>

#include <math.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                         DRIVERS FOR OPTIMIZATION AND NONLINEAR EQUATIONS */

/*--------------------------------------------------------------------------*/
/*                                                                 function */
/* function(tag, m, n, x[n], y[m])                                          */
fint function_(fint* ftag,
               fint* fm,
               fint* fn,
               fdouble* fargument,
               fdouble* fresult) {
    int rc= -1;
    short tag= (short) *ftag;
    int m=*fm,  n=*fn;
    double* argument = myalloc1(n);
    double* result = myalloc1(m);
    spread1(n,fargument,argument);
    rc= function(tag,m,n,argument,result);
    pack1(m,result,fresult);
    myfree1(argument);
    myfree1(result);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 gradient */
/* gradient(tag, n, x[n], g[n])                                             */
fint gradient_(fint* ftag,
               fint* fn,
               fdouble* fargument,
               fdouble* fresult) {
    int rc= -1;
    short tag= (short) *ftag;
    int n=*fn;
    double* argument=myalloc1(n);
    double* result=myalloc1(n);
    spread1(n,fargument,argument);
    rc= gradient(tag,n,argument,result);
    pack1(n,result,fresult);
    myfree1(result);
    myfree1(argument);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* vec_jac(tag, m, n, repeat, x[n], u[m], v[n])                             */
fint vec_jac_(fint* ftag,
              fint* fm,
              fint* fn,
              fint* frepeat,
              fdouble* fargument,
              fdouble* flagrange,
              fdouble* frow) {
    int rc= -1;
    short tag= (short) *ftag;
    int m=*fm, n=*fn, repeat=*frepeat;
    double* argument = myalloc1(n);
    double* lagrange = myalloc1(m);
    double* row = myalloc1(n);
    spread1(m,flagrange,lagrange);
    spread1(n,fargument,argument);
    rc= vec_jac(tag,m,n,repeat,argument,lagrange, row);
    pack1(n,row,frow);
    myfree1(argument);
    myfree1(lagrange);
    myfree1(row);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 jacobian */
/* jacobian(tag, m, n, x[n], J[m][n])                                       */
fint jacobian_(fint* ftag,
               fint* fdepen,
               fint* findep,
               fdouble *fargument,
               fdouble *fjac) {
    int rc= -1;
    short tag= (short) *ftag;
    int depen=*fdepen, indep=*findep;
    double** Jac = myalloc2(depen,indep);
    double* argument = myalloc1(indep);
    spread1(indep,fargument,argument);
    rc= jacobian(tag,depen,indep,argument,Jac);
    pack2(depen,indep,Jac,fjac);
    myfree2(Jac);
    myfree1(argument);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* jac_vec(tag, m, n, x[n], v[n], u[m]);                                    */
fint jac_vec_(fint* ftag,
              fint* fm,
              fint* fn,
              fdouble* fargument,
              fdouble* ftangent,
              fdouble* fcolumn) {
    int rc= -1;
    short tag= (short) *ftag;
    int m=*fm, n=*fn;
    double* argument = myalloc1(n);
    double* tangent = myalloc1(n);
    double* column = myalloc1(m);
    spread1(n,ftangent,tangent);
    spread1(n,fargument,argument);
    rc= jac_vec(tag,m,n,argument,tangent,column);
    pack1(m,column,fcolumn);
    myfree1(argument);
    myfree1(tangent);
    myfree1(column);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* hess_vec(tag, n, x[n], v[n], w[n])                                       */
fint hess_vec_(fint* ftag,
               fint* fn,
               fdouble *fargument,
               fdouble *ftangent,
               fdouble *fresult) {
    int   rc= -1;
    short tag= (short) *ftag;
    int   n=*fn;
    double *argument = myalloc1(n);
    double *tangent = myalloc1(n);
    double *result = myalloc1(n);
    spread1(n,fargument,argument);
    spread1(n,ftangent,tangent);
    rc= hess_vec(tag,n,argument,tangent,result);
    pack1(n,result,fresult);
    myfree1(argument);
    myfree1(tangent);
    myfree1(result);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                  hessian */
/* hessian(tag, n, x[n], lower triangle of H[n][n])                         */
fint hessian_(fint* ftag,
              fint* fn,
              fdouble* fx,
              fdouble* fh) /* length of h should be n*n but the
                            upper half of this matrix remains unchanged */
{
    int rc= -1;
    short tag= (short) *ftag;
    int n=*fn;
    double** H = myalloc2(n,n);
    double* x = myalloc1(n);
    spread1(n,fx,x);
    rc= hessian(tag,n,x,H);
    pack2(n,n,H,fh);
    myfree2(H);
    myfree1(x);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* lagra_hess_vec(tag, m, n, x[n], v[n], u[m], w[n])                        */
fint lagra_hess_vec_(fint* ftag,
                     fint* fm,
                     fint* fn,
                     fdouble *fargument,
                     fdouble *ftangent,
                     fdouble *flagrange,
                     fdouble *fresult) {
    int rc=-1;
    short tag= (short) *ftag;
    int m=*fm, n=*fn;
    double *argument = myalloc1(n);
    double *tangent = myalloc1(n);
    double *lagrange = myalloc1(m);
    double *result = myalloc1(n);
    spread1(n,fargument,argument);
    spread1(n,ftangent,tangent);
    spread1(m,flagrange,lagrange);
    rc= lagra_hess_vec(tag,m,n,argument,tangent,lagrange,result);
    pack1(n,result,fresult);
    myfree1(argument);
    myfree1(tangent);
    myfree1(lagrange);
    myfree1(result);
    return rc;
}

END_C_DECLS
