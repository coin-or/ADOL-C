/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     forward_partx.c
 Revision: $Id$
 Contents: 
 
 Copyright (c) Andrea Walther
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#include <adolc/adalloc.h>
#include <adolc/interfaces.h>

BEGIN_C_DECLS

/*--------------------------------------------------------------------------*/
/*                                                                ZOS_PARTX */
/* zos_forward_partx(tag, m, mdim[n], n, x[n][d], y[m])                     */
/* (based on zos_forward)                                                   */

int zos_forward_partx(short tag, int m, int n, int *ndim, double **x, double *y) {
    double *x0;        /* base point */
    int i,j,ind,sum_n, rc;

    sum_n = 0;
    for(i=0;i<n;i++)
        sum_n += ndim[i];

    x0 = myalloc1(sum_n);

    ind = 0;
    for(i=0;i<n;i++)
        for(j=0;j<ndim[i];j++) {
            x0[ind] = x[i][j];
            ind++;
        }

    rc = zos_forward(tag,m,sum_n,0,x0,y);

    myfree1(x0);

    return rc;
}


/*--------------------------------------------------------------------------*/
/*                                                                FOS_PARTX */
/* fos_forward_partx(tag, m, n, ndim[n], x[n][][2], y[m][2])                */
/* (based on fos_forward)                                                   */

int fos_forward_partx(short tag, int m, int n, int *ndim, double ***x, double **y) {
    double *x0;        /* base point */
    double *xtay;      /* Taylor coefficients */
    double *y0;        /* result */
    double *ytay;      /* derivatives */
    int i,j,ind,sum_n, rc;

    sum_n = 0;
    for(i=0;i<n;i++)
        sum_n += ndim[i];

    x0 = myalloc1(sum_n);
    xtay = myalloc1(sum_n);
    y0 = myalloc1(m);
    ytay = myalloc1(m);

    ind = 0;
    for(i=0;i<n;i++)
        for(j=0;j<ndim[i];j++) {
            x0[ind] = x[i][j][0];
            xtay[ind] = x[i][j][1];
            ind++;
        }

    rc = fos_forward(tag,m,sum_n,0,x0,xtay,y0,ytay);

    for(i=0;i<m;i++) {
        y[i][0] = y0[i];
        y[i][1] = ytay[i];
    }

    myfree1(x0);
    myfree1(xtay);
    myfree1(y0);
    myfree1(ytay);

    return rc;
}


/*--------------------------------------------------------------------------*/
/*                                                                HOS_PARTX */
/* hos_forward_partx(tag, m, n, ndim[n], d, x[n][][d+1], y[m][d+1])         */
/* (based on hos_forward)                                                   */

int hos_forward_partx(short tag, int m, int n, int *ndim, int d, double ***x, double **y) {
    double *x0;        /* base point */
    double **xtay;     /* Taylor coefficients */
    double *y0;        /* result */
    double **ytay;     /* derivaties */
    int i,j,k,ind,sum_n, rc;

    sum_n = 0;
    for(i=0;i<n;i++)
        sum_n += ndim[i];

    x0 = myalloc1(sum_n);
    xtay = myalloc2(sum_n,d);
    y0 = myalloc1(m);
    ytay = myalloc2(m,d);

    ind = 0;
    for(i=0;i<n;i++)
        for(j=0;j<ndim[i];j++) {
            x0[ind] = x[i][j][0];
            for(k=0;k<d;k++)
                xtay[ind][k] = x[i][j][k+1];
            ind++;
        }

    rc = hos_forward(tag,m,sum_n,d,0,x0,xtay,y0,ytay);

    for(i=0;i<m;i++) {
        y[i][0] = y0[i];
        for(j=0;j<d;j++)
            y[i][j+1] = ytay[i][j];
    }

    myfree1(x0);
    myfree2(xtay);
    myfree1(y0);
    myfree2(ytay);

    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                FOV_PARTX */
/* fov_forward_partx(tag, m, n, ndim[n], p, x[n][], X[n][][p],
                     y[m], Y[m][p]) */
/* (based on fov_forward)                                                   */

int fov_forward_partx(short tag, int m, int n, int *ndim,int p,
                      double **x, double ***Xppp, double* y, double **Ypp) {
    double *x0;      /* base point */
    double **X;      /* Taylor coefficients */
    int i,j,k,ind,sum_n, rc;

    sum_n = 0;
    for(i=0;i<n;i++)
        sum_n += ndim[i];

    x0 = myalloc1(sum_n);
    X = myalloc2(sum_n,p);

    ind = 0;
    for(i=0;i<n;i++)
        for(j=0;j<ndim[i];j++) {
            x0[ind] = x[i][j];
            for(k=0;k<p;k++)
                X[ind][k] = Xppp[i][j][k];
            ind++;
        }

    rc = fov_forward(tag,m,sum_n,p,x0,X,y,Ypp);

    myfree1(x0);
    myfree2(X);

    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                HOV_PARTX */
/* hov_forward_partx(tag, m, n, ndim[n], p, x[n][], X[n][][p][d],
                     y[m], Y[m][p][d]) */
/* (based on hov_forward)                                                   */

int hov_forward_partx(short tag, int m, int n, int *ndim, int d, int p,
                      double **x, double ****Xpppp, double* y, double ***Yppp) {
    double *x0;       /* base point */
    double ***X;      /* Taylor coefficients */
    int i,j,k,l,ind,sum_n, rc;

    sum_n = 0;
    for(i=0;i<n;i++)
        sum_n += ndim[i];

    x0 = myalloc1(sum_n);
    X = myalloc3(sum_n,p,d);

    ind = 0;
    for(i=0;i<n;i++)
        for(j=0;j<ndim[i];j++) {
            x0[ind] = x[i][j];
            for(k=0;k<p;k++)
                for(l=0;l<d;l++)
                    X[ind][k][l] = Xpppp[i][j][k][l];
            ind++;

        }

    rc = hov_forward(tag,m,sum_n,d,p,x0,X,y,Yppp);

    myfree1(x0);
    myfree3(X);

    return rc;
}

/****************************************************************************/

END_C_DECLS
