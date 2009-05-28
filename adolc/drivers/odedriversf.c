/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/odedriversf.c
 Revision: $Id$
 Contents: Easy to use drivers for optimization and nonlinear equations
           (Implementation of the Fortran callable interfaces).
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/drivers/odedrivers.h>
#include <adolc/interfaces.h>
#include <adolc/adalloc.h>
#include <adolc/fortutils.h>

#include <math.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                         DRIVERS FOR ODEs */

/*--------------------------------------------------------------------------*/
/*                                                                  forodec */
/* forodec(tag, n, tau, dold, dnew, X[n][d+1])                              */
fint forodec_(fint* ftag,    /* tape identifier */
              fint* fn,       /* space dimension */
              fdouble* ftau,  /* scaling defaults to 1.0 */
              fint* fdol,     /* previous degree defaults to zero */
              fint* fdeg,     /* New degree of consistency        */
              fdouble* fy)    /* Taylor series                    */
{
    int rc= -1;
    short tag= (short) *ftag;
    int n=*fn, dol=*fdol, deg=*fdeg;
    int i;
    double tau=*ftau;
    double** Y = myalloc2(n,deg+1);
    for(i=0;i<n;i++)
        *Y[i] = fy[i];
    rc= forodec(tag,n,tau,dol,deg,Y);
    pack2(n,deg+1,Y,fy);
    free((char*)*Y);
    free((char*)Y);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                  accodec */
/* accodec(n, tau, d, Z[n][n][d+1], B[n][n][d+1], nz[n][n])                 */
fint accodec_(fint* fn,             /* space dimension */
              fdouble* ftau,        /* scaling defaults to 1.0 */
              fint* fdeg,           /* highest degree          */
              fdouble* fa,          /* input tensor of "partial" Jacobians */
              fdouble* fb)          /* output tensor of "total" Jacobians  */
{
    int rc= 1;
    int n=*fn, deg=*fdeg;
    double tau=*ftau;
    double*** A = myalloc3(n,n,deg);
    double*** B = myalloc3(n,n,deg);
    spread3(n,n,deg,fa,A);
    accodec(n,tau,deg,A,B,0);
    pack3(n,n,deg,B,fb);
    free((char*)**A);
    free((char*)*A);
    free((char*)A);
    free((char*)**B);
    free((char*)*B);
    free((char*)B);
    return rc;
}

END_C_DECLS
