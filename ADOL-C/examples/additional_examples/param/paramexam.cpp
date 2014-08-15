/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     paramexam.cpp
 Revision: $Id$
 Contents: example for parameter dependent functions
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adolc.h>

#include <cmath>

#include "myclock.h"

using namespace std;

/****************************************************************************/
/*                                                    CONSTANTS & VARIABLES */
const double TE = 0.01; /* originally 0.0 */
const double R  = sqrt(2.0);


/****************************************************************************/
/*                                                         HELMHOLTZ ENERGY */
adouble energy( int n, adouble x[], double bv[] ) {
    adouble he, xax, bx, tem;
    int i,j;
    xax = 0;
    bx  = 0;
    he  = 0;
    for (i=0; i<n; i++) {
        he += x[i]*log(x[i]);
        bx +=  bv[i]*x[i];
        tem = (2.0/(1.0+i+i))*x[i];
        for (j=0; j<i; j++)
            tem += (1.0/(1.0+i+j))*x[j];
        xax += x[i]*tem;
    }
    xax *= 0.5;
    he   = 1.3625E-3*(he-TE*log(1.0-bx));
    he   = he - log((1+bx*(1+R))/(1+bx*(1-R)))*xax/bx;
    return he;
}

/* Now with parameters */
adouble energy_p( int n, adouble x[], double bv[] ) {
    adouble he, xax, bx, tem;
    int i,j;
    xax = 0;
    bx  = 0;
    he  = 0;
    for (i=0; i<n; i++) {
        he += x[i]*log(x[i]);
        bx +=  mkparam(bv[i])*x[i];
        tem = (2.0/(1.0+i+i))*x[i];
        for (j=0; j<i; j++)
            tem += (1.0/(1.0+i+j))*x[j];
        xax += x[i]*tem;
    }
    xax *= 0.5;
    he   = 1.3625E-3*(he-TE*log(1.0-bx));
    he   = he - log((1+bx*(1+R))/(1+bx*(1-R)))*xax/bx;
    return he;
}

/* now with more independents */
adouble energy_a( int n, adouble x[], adouble b[] ) {
    adouble he, xax, bx, tem;
    int i,j;
    xax = 0;
    bx  = 0;
    he  = 0;
    for (i=0; i<n; i++) {
        he += x[i]*log(x[i]);
        bx +=  b[i]*x[i];
        tem = (2.0/(1.0+i+i))*x[i];
        for (j=0; j<i; j++)
            tem += (1.0/(1.0+i+j))*x[j];
        xax += x[i]*tem;
    }
    xax *= 0.5;
    he   = 1.3625E-3*(he-TE*log(1.0-bx));
    he   = he - log((1+bx*(1+R))/(1+bx*(1-R)))*xax/bx;
    return he;
}

/****************************************************************************/
/*                                                                     MAIN */
/* This program computes first order directional derivatives
   for the helmholtz energy function */
int main() {
    int nf, n, j, l;
    fprintf(stdout,"HELM-AUTO-EXAM (ADOL-C Example)\n\n");
    fprintf(stdout," # of independents/10 =? \n ");
    scanf("%d",&nf);

    /*--------------------------------------------------------------------------*/
    double result = 0.0, result_p = 0.0, result_a;            /* Initilizations */
    double tt1s, tt1e, tt2s, tt2e, trt1s, trt1e, tpxs, tpxe, 
        tt3s, tt3e, tixs, tixe;
    n = 10 * nf;
    double* bv   = new double[n];
    double* grad = new double[n];
    double* grad_p = new double[n];
    double* px = new double[n];
    double* lpx = new double[2*n];
    double* grad_a = new double[2*n];
    double** hess = myalloc2(n,n);
    double** hess_p = myalloc2(n,n);
    double** hess_a = myalloc2(2*n,2*n);

    adouble* x   = new adouble[n];
    adouble* b   = new adouble[n];
    adouble he;

    double r = 1.0/n;
    for (j=0; j<n; j++)
        bv[j]= 0.02*(1.0+fabs(sin(double(j))));

   /*--------------------------------------------------------------------------*/
    int imd_rev = 1;                                     /* Tracing with keep */
    tt1s = myclock();
    trace_on(1,imd_rev);
    for (j=0; j<n; j++)
        x[j] <<= (px[j] = r*sqrt(1.0+j));
    he = energy(n,x,bv);
    he >>= result;
    trace_off();
    reverse(1,1,n,0,1.0,grad);             /* reverse computation of gradient */
    hessian2(1,n,px,hess);
    tt1e = myclock();

   /*--------------------------------------------------------------------------*/
    imd_rev = 1;                                     /* Tracing with keep */
    tt2s = myclock();
    trace_on(2,imd_rev);
    for (j=0; j<n; j++)
        x[j] <<= px[j];
    he = energy_p(n,x,bv);
    he >>= result_p;
    trace_off();
    reverse(2,1,n,0,1.0,grad_p);             /* reverse computation of gradient */
    hessian2(2,n,px,hess_p);
    tt2e = myclock();

    /*--------------------------------------------------------------------------*/
    
    for (j=0; j<n; j++) {
        lpx[j] = px[j];
        lpx[n+j] = bv[j];
    }
    imd_rev = 1;                                     /* Tracing with keep */
    tt3s = myclock();
    trace_on(3,imd_rev);
    for (j=0; j<n; j++) 
        x[j] <<= px[j];
    for (j=0; j<n; j++) 
        b[j] <<= bv[j];
    he = energy_a(n,x,b);
    he >>= result_a;
    trace_off();
    reverse(3,1,2*n,0,1.0,grad_a);             /* reverse computation of gradient */
    hessian2(3,2*n,lpx,hess_a);
    tt3e = myclock();
    /*--------------------------------------------------------------------------*/

    printTapeStats(stdout,1);
    printTapeStats(stdout,2);
    printTapeStats(stdout,3);
    fprintf(stdout, "%14.6E -- energy\n",result);
    fprintf(stdout, "%14.6E -- energy\n",result_p);
    fprintf(stdout, "%14.6E -- energy\n",result_a);

    /*--------------------------------------------------------------------------*/

    for (l=0; l<n; l++)                                            /* results */
        fprintf(stdout,"%3d: 2*%14.6E - %14.6E - %14.6E = %14.6E ( = 0 )\n",l,grad[l],grad_p[l],grad_a[l],2*grad[l]-grad_p[l]-grad_a[l]);

    /*--------------------------------------------------------------------------*/
    /* change constant parameters */
    fprintf(stdout, "changed constant parameters\n");
    for (j=0; j<n; j++)
        bv[j]= 0.01*(1.0+fabs(sin(double(j))));

    /* parameter tape */
    tpxs = myclock();
    set_param_vec(2,n,bv);
    zos_forward(2,1,n,1,px,&result_p);
    reverse(2,1,n,0,1.0,grad_p);
    hessian2(2,n,px,hess_p);
    tpxe = myclock();

    /* double independents tape */
    for (j=0; j<n; j++) {
        lpx[j] = px[j];
        lpx[n+j] = bv[j];
    }
    tixs = myclock();
    zos_forward(3,1,2*n,1,lpx,&result_a);
    reverse(3,1,2*n,0,1.0,grad_a);
    hessian2(3,2*n,lpx,hess_a);
    tixe = myclock();

    /*--------------------------------------------------------------------------*/
    /* with retaping on tape 1 */
    trt1s = myclock();
    trace_on(1,imd_rev);
    for (j=0; j<n; j++)
        x[j] <<= px[j];
    he = energy(n,x,bv);
    he >>= result;
    trace_off();

    reverse(1,1,n,0,1.0,grad);             /* reverse computation of gradient */
    hessian2(1,n,px,hess);
    trt1e = myclock();

    fprintf(stdout, "%14.6E -- energy\n",result);
    fprintf(stdout, "%14.6E -- energy\n",result_p);
    fprintf(stdout, "%14.6E -- energy\n",result_a);
    
    for (l=0; l<n; l++)                                            /* results */
        fprintf(stdout,"%3d: 2*%14.6E - %14.6E - %14.6E = %14.6E ( = 0 )\n",l,grad[l],grad_p[l],grad_a[l],2*grad[l]-grad_p[l]-grad_a[l]);

    fprintf(stdout, "\n\n Times for ");
    fprintf(stdout, "\n Tracing + Reverse + Hess2 1: \t%E", tt1e-tt1s);
    fprintf(stdout, "\n Tracing + Reverse + Hess2 2: \t%E", tt2e-tt2s);
    fprintf(stdout, "\n Tracing + Reverse + Hess2 3: \t%E", tt3e-tt3s);
    fprintf(stdout, "\n Retracing Reverse + Hess2 1: \t%E", trt1e-trt1s);
    fprintf(stdout, "\n Point change grad + Hess2 2: \t%E", tpxe-tpxs);
    fprintf(stdout, "\n Point change grad + Hess2 3: \t%E", tixe-tixs);
    

    myfree2(hess);
    myfree2(hess_p);
    myfree2(hess_a);
    delete[] bv;
    delete[] px;
    delete[] grad;
    delete[] grad_p;
    delete[] lpx;
    delete[] grad_a;
    delete[] x;
    return 0;
}
