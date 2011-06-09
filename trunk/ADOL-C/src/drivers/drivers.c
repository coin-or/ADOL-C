/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/drivers.c
 Revision: $Id$
 Contents: Easy to use drivers for optimization and nonlinear equations
           (Implementation of the C/C++ callable interfaces).
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/
#include <adolc/drivers/drivers.h>
#include <adolc/interfaces.h>
#include <adolc/adalloc.h>

#include <math.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                         DRIVERS FOR OPTIMIZATION AND NONLINEAR EQUATIONS */

/*--------------------------------------------------------------------------*/
/*                                                                 function */
/* function(tag, m, n, x[n], y[m])                                          */
int function(short tag,
             int m,
             int n,
             double* argument,
             double* result) {
    int rc= -1;

    rc= zos_forward(tag,m,n,0,argument,result);

    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 gradient */
/* gradient(tag, n, x[n], g[n])                                             */
int gradient(short tag,
             int n,
             const double* argument,
             double* result) {
    int rc= -1;
    double one = 1.0;

    rc = zos_forward(tag,1,n,1,argument,result);
    if(rc < 0)
        return rc;
    MINDEC(rc, fos_reverse(tag,1,n,&one,result));
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* vec_jac(tag, m, n, repeat, x[n], u[m], v[n])                             */
int vec_jac(short tag,
            int m,
            int n,
            int repeat,
            double* argument,
            double* lagrange,
            double* row) {
    int rc= -1;
    double *y = NULL;

    if(!repeat) {
        y = myalloc1(m);
        rc = zos_forward(tag,m,n,1, argument, y);
        if(rc < 0) return rc;
    }
    MINDEC(rc, fos_reverse(tag,m,n,lagrange,row));
    if (!repeat) myfree1(y);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 jacobian */
/* jacobian(tag, m, n, x[n], J[m][n])                                       */

int jacobian(short tag,
             int depen,
             int indep,
             const double *argument,
             double **jacobian) {
    int rc;
    double *result, **I;

    result = myalloc1(depen);

    if (indep/2 < depen) {
        I = myallocI2(indep);
        rc = fov_forward(tag,depen,indep,indep,argument,I,result,jacobian);
        myfreeI2(indep, I);
    } else {
        I = myallocI2(depen);
        rc = zos_forward(tag,depen,indep,1,argument,result);
        if (rc < 0) return rc;
        MINDEC(rc,fov_reverse(tag,depen,indep,depen,I,jacobian));
        myfreeI2(depen, I);
    }

    myfree1(result);

    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                           large_jacobian */
/* large_jacobian(tag, m, n, k, x[n], y[m], J[m][n])                        */

int large_jacobian(short tag,
		   int depen,
		   int indep,
		   int runns,
		   double *argument,
		   double *result,
		   double **jacobian)
{
    int rc, dirs, i;
    double **I;

	I = myallocI2(indep);
    if (runns > indep) runns = indep;
    if (runns < 1)     runns = 1;
    dirs = indep / runns;
    if (indep % runns) ++dirs;
    for (i=0; i<runns-1; ++i) {
        rc = fov_offset_forward(tag, depen, indep, dirs, i * dirs, argument,
                I, result, jacobian);
    }
    dirs = indep - (runns-1) * dirs;
    rc = fov_offset_forward(tag, depen, indep, dirs, indep-dirs, argument,
		     I, result, jacobian);
    myfreeI2(indep, I);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                  jac_vec */
/* jac_vec(tag, m, n, x[n], v[n], u[m]);                                    */
int jac_vec(short tag,
            int m,
            int n,
            double* argument,
            double* tangent,
            double* column) {
    int rc= -1;
    double *y;

    y = myalloc1(m);

    rc = fos_forward(tag, m, n, 0, argument, tangent, y, column);
    myfree1(y);

    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 hess_vec */
/* hess_vec(tag, n, x[n], v[n], w[n])                                       */
int hess_vec(short tag,
             int n,
             double *argument,
             double *tangent,
             double *result) {
    double one = 1.0;
    return lagra_hess_vec(tag,1,n,argument,tangent,&one,result);
}

/*--------------------------------------------------------------------------*/
/*                                                                 hess_mat */
/* hess_mat(tag, n, q, x[n], V[n][q], W[n][q])                              */
int hess_mat(short tag,
             int n,
             int q,
             double *argument,
             double **tangent,
             double **result) {
    int rc;
    int i,j;
    double y;
    double*** Xppp;
    double*** Yppp;
    double*** Zppp;
    double**  Upp;

    Xppp = myalloc3(n,q,1);   /* matrix on right-hand side  */
    Yppp = myalloc3(1,q,1);   /* results of hos_wk_forward  */
    Zppp = myalloc3(q,n,2);   /* result of Up x H x XPPP */
    Upp  = myalloc2(1,2);     /* vector on left-hand side */

    for (i = 0; i < n; ++i)
        for (j = 0; j < q; ++j)
            Xppp[i][j][0] = tangent[i][j];

    Upp[0][0] = 1;
    Upp[0][1] = 0;

    rc = hov_wk_forward(tag, 1, n, 1, 2, q, argument, Xppp, &y, Yppp);
    MINDEC(rc, hos_ov_reverse(tag, 1, n, 1, q, Upp, Zppp));

    for (i = 0; i < q; ++i)
        for (j = 0; j < n; ++j)
            result[j][i] = Zppp[i][j][1];

    myfree2(Upp);
    myfree3(Zppp);
    myfree3(Yppp);
    myfree3(Xppp);

    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                  hessian */
/* hessian(tag, n, x[n], lower triangle of H[n][n])                         */
/* uses Hessian-vector product                                              */
int hessian(short tag,
            int n,
            double* argument,
            double** hess) {
    int rc= 3;
    int i,j;
    double *v = myalloc1(n);
    double *w = myalloc1(n);
    for(i=0;i<n;i++) v[i] = 0;
    for(i=0;i<n;i++) {
        v[i] = 1;
        MINDEC(rc, hess_vec(tag, n, argument, v, w));
        if( rc < 0) {
            free((char *)v);
            free((char *) w);
            return rc;
        }
        for(j=0;j<=i;j++)
            hess[i][j] = w[j];
        v[i] = 0;
    }

    free((char *)v);
    free((char *) w);
    return rc;
    /* Note that only the lower triangle of hess is filled */
}

/*--------------------------------------------------------------------------*/
/*                                                                 hessian2 */
/* hessian2(tag, n, x[n], lower triangle of H[n][n])                        */
/* uses Hessian-matrix product                                              */
int hessian2(short tag,
             int n,
             double* argument,
             double** hess) {
    int rc;
    int i,j;

    double*** Xppp = myalloc3(n,n,1);   /* matrix on right-hand side  */
    double*   y    = myalloc1(1);       /* results of function evaluation */
    double*** Yppp = myalloc3(1,n,1);   /* results of hos_wk_forward  */
    double*** Zppp = myalloc3(n,n,2);   /* result of Up x H x XPPP */
    double**  Upp  = myalloc2(1,2);     /* vector on left-hand side */

    for (i=0; i<n; i++) {
        for (j=0;j<n;j++)
            Xppp[i][j][0] = 0;
        Xppp[i][i][0] = 1;
    }

    Upp[0][0] = 1;
    Upp[0][1] = 0;

    rc = hov_wk_forward(tag,1,n,1,2,n,argument,Xppp,y,Yppp);
    MINDEC(rc,hos_ov_reverse(tag,1,n,1,n,Upp,Zppp));

    for (i=0; i<n; i++)
        for (j=0;j<=i;j++)
            hess[i][j] = Zppp[i][j][1];

    myfree2(Upp);
    myfree3(Zppp);
    myfree3(Yppp);
    myfree1(y);
    myfree3(Xppp);
    return rc;
    /* Note that only the lower triangle of hess is filled */
}

/*--------------------------------------------------------------------------*/
/*                                                           lagra_hess_vec */
/* lagra_hess_vec(tag, m, n, x[n], v[n], u[m], w[n])                        */
int lagra_hess_vec(short tag,
                   int m,
                   int n,
                   double *argument,
                   double *tangent,
                   double *lagrange,
                   double *result) {
    int rc=-1;
    int i;
    int degree = 1;
    int keep = degree+1;
    double **X, *y, *y_tangent;

    X = myalloc2(n,2);
    y = myalloc1(m);
    y_tangent = myalloc1(m);

    rc = fos_forward(tag, m, n, keep, argument, tangent, y, y_tangent);

    if(rc < 0) return rc;

    MINDEC(rc, hos_reverse(tag, m, n, degree, lagrange, X));

    for(i = 0; i < n; ++i)
        result[i] = X[i][1];

    myfree1(y_tangent);
    myfree1(y);
    myfree2(X);

    return rc;
}

END_C_DECLS
