/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse_hessian.cpp
 Revision: $Id$
 Contents: example for computation of sparse hessians

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

#include <math.h>
#include <cstdlib>
#include <cstdio>

#include <adolc/adolc.h>
#include <adolc/adolc_sparse.h>

#define tag 1

double  feval(double *x);
adouble feval_ad(adouble *x);

void printmat(const char* kette, int n, int m, double** M);

int main() {
    int n=6;
    double f, x[6];
    adouble fad, xad[6];

    int i, j;

/****************************************************************************/
/*******                function evaluation                   ***************/
/****************************************************************************/

    for(i=0;i<n;i++)
        x[i] = log(1.0+i);

    /* Tracing of function f(x) */

    trace_on(tag);
      for(i=0;i<n;i++)
        xad[i] <<= x[i];

      fad = feval_ad(xad);

      fad >>= f;
    trace_off();

    printf("\n f = %e\n\n\n",f);


/****************************************************************************/
/********           For comparisons: Full Hessian                    ********/
/****************************************************************************/

    double **H;
    H = myalloc2(n,n);

    hessian(tag,n,x,H);

    printmat(" H",n,n,H);
    printf("\n");


/****************************************************************************/
/*******       sparse Hessians, complete driver              ***************/
/****************************************************************************/

    /* coordinate format for Hessian */
    unsigned int    *rind  = NULL;
    unsigned int    *cind  = NULL;
    double *values = NULL;
    int nnz;
    int options[2];

    options[0] = 0;          /*                               safe mode (default) */ 
    options[1] = 0;          /*                       indirect recovery (default) */ 

    sparse_hess(tag, n, 0, x, &nnz, &rind, &cind, &values, options);

    printf("In sparse format:\n");
    for (i=0;i<nnz;i++)
        printf("%2d %2d %10.6f\n\n",rind[i],cind[i],values[i]);

    free(rind); rind = NULL;
    free(cind); cind = NULL;
    free(values); values = NULL;

    options[0] = 0;          /*                               safe mode (default) */ 
    options[1] = 1;          /*                                   direct recovery */ 

    sparse_hess(tag, n, 0, x, &nnz, &rind, &cind, &values, options);

    printf("In sparse format:\n");
    for (i=0;i<nnz;i++)
        printf("%2d %2d %10.6f\n\n",rind[i],cind[i],values[i]);

    free(rind); rind=NULL;
    free(cind); cind=NULL;
    free(values); values=NULL;

/*--------------------------------------------------------------------------*/
/*  change value of x, but not the sparsity pattern                         */
/*--------------------------------------------------------------------------*/

    for(i=0;i<n;i++)
        x[i] = 2.0*i;

/*  For comparisons: Full Hessian:                                         */

    hessian(tag,n,x,H);

    printmat(" H",n,n,H);
    printf("\n");

/*  repeated call of sparse_hess with same sparsity pattern => repeat = 1 */

    sparse_hess(tag, n, 0, x, &nnz, &rind, &cind, &values, options);

    printf("In sparse format:\n");
    for (i=0;i<nnz;i++)
        printf("%2d %2d %10.6f\n\n",rind[i],cind[i],values[i]);

    free(rind); rind=NULL;
    free(cind); cind=NULL;
    free(values); values=NULL;

/****************************************************************************/
/*******        sparse Hessians, separate drivers             ***************/
/****************************************************************************/

/*--------------------------------------------------------------------------*/
/*                                                 sparsity pattern Hessian */
/*--------------------------------------------------------------------------*/


    unsigned int  **HP=NULL;                /* compressed block row storage */
    int ctrl;

    HP = (unsigned int **) malloc(n*sizeof(unsigned int*));
    ctrl = 0;

    hess_pat(tag, n, x, HP, ctrl);

    printf("\n");
    printf("Sparsity pattern of Hessian: \n");
    for (i=0;i<n;i++) {
        printf(" %d: ",i);
        for (j=1;j<= (int) HP[i][0];j++)
            printf(" %d ",HP[i][j]);
        printf("\n");
    }
    printf("\n");


/*--------------------------------------------------------------------------*/
/*                                                              seed matrix */
/*--------------------------------------------------------------------------*/

    double **Seed;
    int p;
    int option = 1;

    /* option = 0  indirect recovery (default), 
       option = 1  direct recovery                       */
 
    generate_seed_hess(n, HP, &Seed, &p, option);

    printmat(" Seed matrix",n,p,Seed);
    printf("\n");

/*--------------------------------------------------------------------------*/
/*                                                       compressed Hessian */
/*--------------------------------------------------------------------------*/

    double **Hcomp;
    Hcomp = myalloc2(n,p);

    hess_mat(tag, n, p, x, Seed, Hcomp);

    printmat("compressed H:",n,p,Hcomp);
    printf("\n");

/*--------------------------------------------------------------------------*/
/*  change value of x, but not the sparsity pattern                         */
/*--------------------------------------------------------------------------*/

    for(i=0;i<n;i++)
        x[i] = 2.0*i;

/*  For comparisons: Full Hessian                                           */

    hessian(tag,n,x,H);

    printmat(" H",n,n,H);
    printf("\n");

    hess_mat(tag, n, p, x, Seed, Hcomp);

    printmat("compressed H:",n,p,Hcomp);
    printf("\n");

    for(i=0;i<n;i++)
       free(HP[i]);
    free(HP);

    myfree2(H);
    myfree2(Hcomp);

    for (i = 0; i < n; i++)
        delete[] Seed[i];
    delete[] Seed;

}


/***************************************************************************/

double feval(double *x) {
    double res;

    res = 0.5*(x[0] - 1)*(x[0] -1) + 0.8*(x[1] - 2)*(x[1] -2)  + 0.9*(x[2] - 3)*(x[2] -3);
    res += 5*x[0]*x[1];
    res += cos(x[3]);
    res += sin(x[4])*pow(x[1],2);
    res += exp(x[5])*x[2];
    res += sin(x[4]*x[5]);

    return res;
}

/***************************************************************************/

adouble feval_ad(adouble *x) {
    adouble res;

    res = 0.5*(x[0] - 1)*(x[0] -1) + 0.8*(x[1] - 2)*(x[1] -2)  + 0.9*(x[2] - 2)*(x[2] -2);
    res += 5*x[0]*x[1];
    res += cos(x[3]);
    res += sin(x[4])*x[1]*x[1];
    res += exp(x[5])*x[2];
    res += sin(x[4]*x[5]);

    return res;
}

/***************************************************************************/

void printmat(const char* name, int m, int n, double** M) {
    int i,j;

    printf("%s \n",name);
    for(i=0; i<m ;i++) {
        printf("\n %d: ",i);
        for(j=0;j<n ;j++)
            printf(" %10.4f ", M[i][j]);
    }
    printf("\n");
}
