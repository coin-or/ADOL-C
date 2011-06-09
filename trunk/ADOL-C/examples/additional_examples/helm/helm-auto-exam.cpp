/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     helm-auto-exam.cpp
 Revision: $Id$
 Contents: example for  Helmholtz energy example 
           Computes gradient using AD driver reverse(..)

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <math.h>


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
    double result = 0.0;                                    /* Initilizations */
    n = 10 * nf;
    double* bv   = new double[n];
    double* grad = new double[n];

    adouble* x   = new adouble[n];
    adouble he;

    double r = 1.0/n;
    for (j=0; j<n; j++)
        bv[j]= 0.02*(1.0+fabs(sin(double(j))));

    /*--------------------------------------------------------------------------*/
    int imd_rev = 1;                                     /* Tracing with keep */
    trace_on(1,imd_rev);
    for (j=0; j<n; j++)
        x[j] <<= r*sqrt(1.0+j);
    he = energy(n,x,bv);
    he >>= result;
    trace_off();
    fprintf(stdout, "%14.6E -- energy\n",result);

    /*--------------------------------------------------------------------------*/
    reverse(1,1,n,0,1.0,grad);             /* reverse computation of gradient */

    /*--------------------------------------------------------------------------*/
    for (l=0; l<n; l++)                                            /* results */
        fprintf(stdout,"%3d: %14.6E,  \n",l,grad[l]);
    fprintf(stdout,"%14.6E -- energy\n",result);

    delete [] x;
    delete [] bv;
    delete [] grad;

    return 1;
}


/****************************************************************************/
/*                                                               THAT'S ALL */

