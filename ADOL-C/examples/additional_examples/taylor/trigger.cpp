/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     trigger.cpp
 Revision: $Id$
 Contents:  Test driver 'inverse_tensor_eval(..)' that allows to 
            compute higher order derivatives of inverse functions
           
            Function model: trigger circuit
 
 Copyright (c) Andrea Walther, Andreas Griewank
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <cstdlib>


/****************************************************************************/
/*                                                                     MAIN */
int main() {
    int i,j,n,d,p,dim;

    /*--------------------------------------------------------------------------*/
    printf(" TRIGGER CIRCUIT EXAMPLE (ADOL-C Example)\n\n");        /* inputs */
    printf(" # of indeps = 7, # of deps = 7  (fixed)\n");

    n = 7;
    p = 3;
    d = 4;

    /*--------------------------------------------------------------------------*/
    double* xp = new double[n];                      /* allocations and inits */
    double* Fhp = new double[n];
    double** S = new double*[n];
    double** tensors;
    int* multi = new int[d];
    int* add = new int[5];

    for(i=0;i<n;i++) {
        S[i] = new double[p];
        for(j=0;j<p;j++)
            S[i][j] = 0;
    }
    S[4][0] = 1;
    S[5][1] = 1;
    S[6][2] = 1;

    xp[0] = 5.155103392445;
    xp[1] = 5.808134401609;
    xp[2] = 5.764511314735;
    xp[3] = 5.569172054801;
    xp[4] = 5.786953999546;
    xp[5] = 1.1294023667;
    xp[6] = 0;

    /*--------------------------------------------------------------------------*/
    trace_on(1);                                      /* tracing the function */
    adouble* x = new adouble[n];
    adouble* Fh = new adouble[n];

    for(i=0;i<n;i++)
        x[i] <<= xp[i];

    Fh[0]=x[0]/20-1e-9*(exp(30*(x[4]-x[0]))+exp(30*(x[2]-x[0])))-1.88e-7;
    Fh[0]+=+.95e-7*(exp(30*(x[4]-x[1]))+exp(30*(x[2]-x[3])));
    Fh[1]= .99e-9*exp(30*(x[4]-x[0]))-1e-7*exp(30*(x[4]-x[1]))+.9901e-7;
    Fh[1]+=+(x[1]-x[2])/50-(6-x[1])/x[5]-x[6];
    Fh[2]= 1e-11*exp(30*(x[2]-x[0]))+5e-9*exp(30*(x[2]-x[3]))-5.01e-9;
    Fh[2]+=-(x[1]-x[2])/50;
    Fh[3]=.99e-9*exp(30*(x[2]-x[0]))-1e-7*exp(30*(x[2]-x[3]))+.9901e-7;
    Fh[3]+=-(6-x[3])/5;
    Fh[4]= x[0]-5.155103392445;
    Fh[5]= x[4]-5.786953999546;
    Fh[6] = x[5]-1.1294023667;

    for(i=0;i<n;i++)
        Fh[i] >>= Fhp[i];
    trace_off();

    /*--------------------------------------------------------------------------*/
    printf("\n  Fh(x,0) = \n (");                      /* inverse_tensor_eval */
    for(i=0;i<n;i++)
        printf(" %f",Fhp[i]);
    printf(" %15.10f )\n\n",Fhp[n-1]);
    dim = binomi(p+d,d);
    tensors = myalloc2(n,dim);

    inverse_tensor_eval(1,n,d,p,xp,tensors,S);

    for(i=0;i<d;i++)
        multi[i] = 0;
    multi[d-1] = 1;
    add[0] = tensor_address(d,multi);
    multi[d-2] = 1;
    add[1] = tensor_address(d,multi);
    multi[d-3] = 1;
    add[2] = tensor_address(d,multi);
    multi[d-1] = 2;
    multi[d-2] = 0;
    multi[d-3] = 0;
    add[3] = tensor_address(d,multi);
    multi[d-1] = 3;
    add[4] = tensor_address(d,multi);
    printf(" Results: \n");
    printf(" Fh^{-1}(0,z) = \n (");
    for(i=0;i<n-1;i++)
        printf(" %15.10f ,",tensors[i][0]);
    printf(" %15.10f )\n\n",tensors[n-1][0]);
    printf(" Fh^{-1}_{z_1}(0,z) = \n (");
    for(i=0;i<n-1;i++)
        printf(" %15.10f ,",tensors[i][add[0]]);
    printf(" %15.10f )\n\n",tensors[n-1][add[0]]);
    printf(" Fh^{-1}_{x_1 x_1}(0,z) = \n (");
    for(i=0;i<n-1;i++)
        printf(" %15.10f ,",tensors[i][add[1]]);
    printf(" %15.10f )\n\n",tensors[n-1][add[1]]);
    printf(" Fh^{-1}_{x_1 x_1 x_1}(0,z) = \n (");
    for(i=0;i<n-1;i++)
        printf(" %15.10f ,",tensors[i][add[2]]);
    printf(" %15.10f )\n\n",tensors[n-1][add[2]]);
    printf(" Fh^{-1}_{x_2}(0,z) = \n (");
    for(i=0;i<n-1;i++)
        printf(" %15.10f ,",tensors[i][add[3]]);
    printf(" %15.10f )\n\n",tensors[n-1][add[3]]);
    printf(" Fh^{-1}_{x_3}(0,z) = \n (");
    for(i=0;i<n-1;i++)
        printf(" %15.10f ,",tensors[i][add[4]]);
    printf(" %15.10f )\n",tensors[n-1][add[4]]);

    return 1;
}

/****************************************************************************/
/*                                                               THAT'S ALL */


