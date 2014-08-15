/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     interfaces.cpp
 Revision: $Id$
 Contents: Genuine C++ Interfaces to ADOL-C forward & reverse calls.
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#include <adolc/interfaces.h>
#include <adolc/adalloc.h>
#include "dvlparms.h"

/****************************************************************************/
/*                                                                   MACROS */
#define fabs(x) ((x) > 0 ? (x) : -(x))
#define ceil(x) ((int)((x)+1) - (int)((x) == (int)(x)))

extern "C" void adolc_exit(int errorcode, const char *what, const char* function, const char *file, int line);

/****************************************************************************/
/*                                           FORWARD MODE, overloaded calls */

/****************************************************************************/
/*                                                             general call */
/*                                                                          */
int forward( short  tag,
             int    m,
             int    n,
             int    d,
             int    keep,
             double **X,
             double **Y)
/* forward(tag, m, n, d, keep, X[n][d+1], Y[m][d+1])                        */
{ /* olvo 980729 general ec */
    static double *x, *y, *xp, *yp;
    static int maxn, maxm;
    int rc = -1, i, k;
    if (n > maxn) {
        if (x)
            myfree1(x);
        if (xp)
            myfree1(xp);
        x  = myalloc1(maxn = n);
        xp = myalloc1(maxn);
    }
    if (m > maxm) {
        if (y)
            myfree1(y);
        if (yp)
            myfree1(yp);
        y  = myalloc1(maxm = m);
        yp = myalloc1(maxm);
    }

    /*------------------------------------------------------------------------*/
    /* prepare input */
    for (i=0; i<n; i++) {
        x[i] = X[i][0];
        if (d == 1)
            xp[i] = X[i][1];
        else
            for (k=0; k<d; k++)
                X[i][k] = X[i][k+1];
    }

    /*------------------------------------------------------------------------*/
    /* function calls */
    if (d == 0)
        rc = zos_forward(tag,m,n,keep,x,y);
    else
        if (d == 1)
            rc = fos_forward(tag,m,n,keep,x,xp,y,yp);
        else
            rc = hos_forward(tag,m,n,d,keep,x,X,y,Y);

    /*------------------------------------------------------------------------*/
    /* prepare output */
    for (i=0; i<n; i++)
        if (d > 1) {
            for (k=d; k>0; k--)
                X[i][k] = X[i][k-1];
            X[i][0] = x[i];
        }

    for (i=0; i<m; i++) {
        if (d == 1)
            Y[i][1] = yp[i];
        else
            for (k=d; k>0; k--)
                Y[i][k] = Y[i][k-1];
        Y[i][0] = y[i];
    }

    return rc;
}


/****************************************************************************/
/*         Y can be one dimensional if m=1                                  */
/*                                                                          */
int forward( short  tag,
             int    m,
             int    n,
             int    d,
             int    keep,
             double **X,
             double *Y)
/* forward(tag, 1, n, d, keep, X[n][d+1], Y[d+1]), m=1                      */
{ /* olvo 980729 general ec */
    static double *x, *xp;
    static int maxn;
    double y;
    int rc= -1, i, k;

    if (m == 1) {
        if (n > maxn) {
            if (x)
                myfree1(x);
            if (xp)
                myfree1(xp);
            x  = myalloc1(maxn = n);
            xp = myalloc1(maxn);
        }

        /*----------------------------------------------------------------------*/
        /* prepare input */
        for (i=0; i<n; i++) {
            x[i] = X[i][0];
            if (d == 1)
                xp[i] = X[i][1];
            else
                for (k=0; k<d; k++)
                    X[i][k] = X[i][k+1];
        }

        /*----------------------------------------------------------------------*/
        /* function calls */
        if (d == 0)
            rc = zos_forward(tag,m,n,keep,x,&y);
        else
            if (d == 1)
                rc = fos_forward(tag,m,n,keep,x,xp,&y,Y);
            else
                rc = hos_forward(tag,m,n,d,keep,x,X,&y,&Y);

        /*----------------------------------------------------------------------*/
        /* prepare output */
        for (i=0; i<n; i++)
            if (d > 1) {
                for (k=d; k>0; k--)
                    X[i][k] = X[i][k-1];
                X[i][0] = x[i];
            }

        for (k=d; k>0; k--)
            Y[k] = Y[k-1];
        Y[0] = y;
    } else {
        fprintf(DIAG_OUT,"ADOL-C error: wrong Y dimension in forward \n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }

    return rc;
}


/****************************************************************************/
/*         X and Y can be one dimensional if d = 0                          */
/*                                                                          */
int forward( short  tag,
             int    m,
             int    n,
             int    d,
             int    keep,
             double *X,
             double *Y)
/* forward(tag, m, n, 0, keep, X[n], Y[m]), d=0                             */
{ int rc = -1;

    if (d != 0) {
        fprintf(DIAG_OUT,"ADOL-C error:  wrong X and Y dimensions in forward \n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    } else
        rc = zos_forward(tag,m,n,keep,X,Y);

    return rc;
}


/****************************************************************************/
/*         X and Y can be one dimensional if d omitted                      */
/*                                                                          */
int forward(short  tag,
            int    m,
            int    n,
            int    keep,
            double *X,
            double *Y)
/* forward(tag, m, n, keep, X[n], Y[m])                                     */
{ return zos_forward(tag,m,n,keep,X,Y);
}


/****************************************************************************/
/*                                                             general call */
/*                                                                          */
int forward( short  tag,
             int    m,
             int    n,
             int    d,
             int    p,
             double *x,
             double ***X,
             double *y,
             double ***Y)
/* forward(tag, m, n, d, p, x[n], X[n][p][d], y[m], Y[m][p][d])             */
{ return hov_forward(tag,m,n,d,p,x,X,y,Y);
}


/****************************************************************************/
/*                                                             general call */
/*                                                                          */
int forward( short  tag,
             int    m,
             int    n,
             int    p,
             double *x,
             double **X,
             double *y,
             double **Y)
/* forward(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p])                      */
{ return fov_forward(tag,m,n,p,x,X,y,Y);
}


/****************************************************************************/
/*                                           REVERSE MODE, overloaded calls */

/****************************************************************************/
/*                                                             general call */
/*                                                                          */
int reverse( short  tag,
             int    m,
             int    n,
             int    d,
             double *u,
             double **Z)
/* reverse(tag, m, n, d, u[m], Z[n][d+1])                                   */
{ return hos_reverse(tag,m,n,d,u,Z);
}


/****************************************************************************/
/*         u can be a scalar if m=1                                         */
/*                                                                          */
int reverse( short  tag,
             int    m,
             int    n,
             int    d,
             double u,
             double **Z)
/* reverse(tag, 1, n, 0, u, Z[n][d+1]), m=1 => u scalar                     */
{ int rc=-1;

    if (m != 1) {
        fprintf(DIAG_OUT,"ADOL-C error:  wrong u dimension in scalar-reverse \n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    } else
        rc = hos_reverse(tag,m,n,d,&u,Z);

    return rc;
}


/****************************************************************************/
/*         Z can be vector if d = 0; Done by specialized code               */
/*                                                                          */
int reverse( short  tag,
             int    m,
             int    n,
             int    d,
             double *u,
             double *Z)
/* reverse(tag, m, n, 0, u[m], Z[n]), d=0                                   */
{ if (d != 0) {
        fprintf(DIAG_OUT,"ADOL-C error:  wrong Z dimension in scalar-reverse \n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }

    return fos_reverse(tag,m,n,u,Z);
}


/****************************************************************************/
/*         u and Z can be scalars if m=1 and d=0;                           */
/*                                                                          */
int reverse( short  tag,
             int    m,
             int    n,
             int    d,
             double u,
             double *Z)
/* reverse(tag, 1, n, 0, u, Z[n]), m=1 and d=0 => u and Z scalars           */
{ int rc=-1;

    if (m != 1 || d != 0 ) {
        fprintf(DIAG_OUT,"ADOL-C error:  wrong u or Z dimension in scalar-reverse \n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    } else
        rc = fos_reverse(tag,m,n,&u,Z);
    \
    return rc;
}


/****************************************************************************/
/*                                                             general call */
/*                                                                          */
int reverse( short  tag,
             int    m,
             int    n,
             int    d,
             int    q,
             double **U,
             double ***Z,
             short  **nz)
/* reverse(tag, m, n, d, q, U[q][m], Z[q][n][d+1], nz[q][n])                */
{ return hov_reverse(tag,m,n,d,q,U,Z,nz);
}


/****************************************************************************/
/*         U can be a vector if m=1                                         */
/*                                                                          */
int reverse( short  tag,
             int    m,
             int    n,
             int    d,
             int    q,
             double *U,
             double ***Z,
             short  **nz)
/* reverse(tag, 1, n, d, q, U[q], Z[q][n][d+1], nz[q][n]), m=1 => u vector  */
{ int rc=-1;

    if (m != 1) {
        fprintf(DIAG_OUT,"ADOL-C error:  wrong U dimension in vector-reverse \n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    } else { /* olvo 980727 ??? */
        /* double** upp = new double*[nrows]; */
        double **upp = (double**) malloc(q*sizeof(double*));
        for (int i=0; i<q; i++)
            upp[i] = &U[i];
        rc=hov_reverse(tag,m,n,d,q,upp,Z,nz);
        /* delete[] upp; */
        free((char*)upp);
    }
    return rc;
}


/****************************************************************************/
/*                                                                          */
/*         If d=0 then Z may be matrix; Done by specialized code            */
/*                                                                          */
int reverse( short  tag,
             int    m,
             int    n,
             int    d,
             int    q,
             double **U,
             double **Z)
/* reverse(tag, m, n, 0, q, U[q][m], Z[q][n]), d=0 => Z matrix              */
{ int rc=-1;

    if (d != 0) {
        fprintf(DIAG_OUT,"ADOL-C error:  wrong degree in vector-reverse \n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    } else
        rc = fov_reverse(tag,m,n,q,U,Z);

    return rc;
}


/****************************************************************************/
/*                                                                          */
/*         d=0 may be omitted, then Z may be a matrix; specialized code     */
/*                                                                          */
int reverse( short  tag,
             int    m,
             int    n,
             int    q,
             double **U,
             double **Z)
/* reverse(tag, m, n, q, U[q][m], Z[q][n]), d=0 => Z matrix                 */
{ int rc=-1;

    rc = fov_reverse(tag,m,n,q,U,Z);

    return rc;
}


/****************************************************************************/
/*                                                                          */
/*         If m=1 and d=0 then U can be vector and Z a matrix but no nz.    */
/*                                                                          */
int reverse( short  tag,
             int    m,
             int    n,
             int    d,
             int    q,
             double *U,
             double **Z)
/* reverse(tag, 1, n, 0, q, U[q], Z[q][n]),
                            m=1 and d=0 => U vector and Z matrix but no nz  */
{ int rc=-1;

    /* olvo 981126 ??? what's that: */
    /* (++d)--; *//* degre is reserved for the future use. Ingore this line */

    if ((m != 1) || (d != 0)) {
        fprintf(DIAG_OUT,"ADOL-C error:  wrong U dimension in vector-reverse \n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    } else { /* olvo 980727 ??? */
        /* double ** upp = new double*[nrows]; */
        double **upp = (double**) malloc(q*sizeof(double*));
        for (int i=0; i<q; i++)
            upp[i] = &U[i];
        rc = fov_reverse(tag,m,n,q,upp,Z);
        /* delete[] upp; */
        free((char*) upp);
    }

    return rc;
}


/****************************************************************************/
/*                                                                          */
/*         If p and U are omitted they default to m and I so that as above  */
/*                                                                          */
int reverse( short  tag,
             int    m,
             int    n,
             int    d,
             double ***Z,
             short  **nz)
/* reverse(tag, m, n, d, Z[p][n][d+1], nz[p][n]),
           If p and U are omitted they default to m and I                   */
{ static int depax;
    static double** I;
    if (m adolc_compsize depax) {
        if (depax)
            myfreeI2(depax,I);
        I = myallocI2(depax = m);
    }
    return hov_reverse(tag,m,n,d,m,I,Z,nz);
}

/****************************************************************************/
/*                                                               THAT'S ALL */
