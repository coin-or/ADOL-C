/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/psdrivers.c
 Revision: $Id$
 Contents: Easy to use drivers for piecewise smooth functions
           (with C and C++ callable interfaces including Fortran 
            callable versions).

 Copyright (c) Andrea Walther, Sabrina Fiege, Kshitij Kulshreshtha 

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduct ion, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
  
----------------------------------------------------------------------------*/

#include <adolc/drivers/psdrivers.h>
#include <adolc/adalloc.h>
#include <adolc/fortutils.h>

#if !defined(ADOLC_NO_MALLOC)
#   define ADOLC_CALLOC(n,m) calloc(n,m)
#else
#   define ADOLC_CALLOC(n,m) rpl_calloc(n,m)
#endif
#if defined(ADOLC_USE_CALLOC)
#  if !defined(ADOLC_NO_MALLOC)
#     define ADOLC_MALLOC(n,m) calloc(n,m)
#  else
#     define ADOLC_MALLOC(n,m) rpl_calloc(n,m)
#  endif
#else
#  if !defined(ADOLC_NO_MALLOC)
#     define ADOLC_MALLOC(n,m) malloc(n*m)
#  else
#     define ADOLC_MALLOC(n,m) rpl_malloc(n*m)
#  endif
#endif

BEGIN_C_DECLS

/****************************************************************************/
/*                                                 DRIVERS FOR PS FUNCTIONS */

/*--------------------------------------------------------------------------*/
/*                                                          abs-normal form */
/* abs_normal(tag,m,n,s,x[n],sig[s],y[m],z[s],cz[s],cy[m],                  */
/*            J[m][n],Y[m][s],Z[s][n],L[s][s])                              */
fint abs_normal_(fint* ftag,
                 fint* fdepen,
                 fint* findep,
                 fint* fswchk,
                 fdouble* fx,
                 fdouble* fy,
                 fdouble* fz,
                 fdouble* fcz,
                 fdouble* fcy,
                 fdouble* fJ,
                 fdouble* fY,
                 fdouble* fZ,
                 fdouble* fL) {
    int rc = -1; 
    short tag = (short)*ftag;
    int m = (int)*fdepen, n = (int)*findep, s=(int)*fswchk;
    double **J, **Y, **Z, **L;
    double *cy, *cz, *x, *y, *z;
    J = myalloc2(m,n);
    Y = myalloc2(m,s);
    Z = myalloc2(s,n);
    L = myalloc2(s,s);
    cy = myalloc1(m);
    cz = myalloc1(s);
    x = myalloc1(n);
    y = myalloc1(m);
    z = myalloc1(s);
    spread1(n,fx,x);
    rc = abs_normal(tag,m,n,s,x,y,z,cz,cy,J,Y,Z,L);
    pack1(m,y,fy);
    pack1(s,z,fz);
    pack1(s,cz,fcz);
    pack1(m,cy,fcy);
    pack2(m,n,J,fJ);
    pack2(m,s,Y,fY);
    pack2(s,n,Z,fZ);
    pack2(s,s,L,fL);
    myfree2(J);
    myfree2(Y);
    myfree2(Z);
    myfree2(L);
    myfree1(x);
    myfree1(y);
    myfree1(z);
    myfree1(cz);
    myfree1(cy);
    return rc;
}

/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS
