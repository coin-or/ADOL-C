/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse_jacobian.cpp
 Revision: $Id$
 Contents: example for computation of sparse jacobians

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

void   ceval_ad(adouble *x, adouble *c);
void   ceval(double *x, double *c);

void printmat(const char* name, int n, int m, double** M);

int main() {
    int n=6, m=3;
    double x[6], c[3];
    adouble xad[6], cad[3];

    int i, j;

/****************************************************************************/
/*******                function evaluation                   ***************/
/****************************************************************************/

    for(i=0;i<n;i++)
        x[i] = log(1.0+i);

    /* Tracing of function c(x) */

    trace_on(tag);
      for(i=0;i<n;i++)
        xad[i] <<= x[i];

      ceval_ad(xad,cad);

      for(i=0;i<m;i++)
        cad[i] >>= c[i];
    trace_off();

    printf("\n c =  ");
    for(j=0;j<m;j++)
        printf(" %e ",c[j]);
    printf("\n");

/****************************************************************************/
/********           For comparisons: Full Jacobian                   ********/
/****************************************************************************/

    double **J;
    J = myalloc2(m,n);

    jacobian(tag,m,n,x,J);

    printmat(" J",m,n,J);
    printf("\n");


/****************************************************************************/
/*******       sparse Jacobians, complete driver              ***************/
/****************************************************************************/

    /* coordinate format for Jacobian */
    unsigned int *rind  = NULL;        /* row indices    */
    unsigned int *cind  = NULL;        /* column indices */
    double       *values = NULL;       /* values         */
    int nnz;
    int options[4];

    options[0] = 0;          /* sparsity pattern by index domains (default) */ 
    options[1] = 0;          /*                         safe mode (default) */ 
    options[2] = 0;          /*              not required if options[0] = 0 */ 
    options[3] = 0;          /*                column compression (default) */ 

    sparse_jac(tag, m, n, 0, x, &nnz, &rind, &cind, &values, options);

    printf("In sparse format:\n");
    for (i=0;i<nnz;i++)
        printf("%2d %2d %10.6f\n\n",rind[i],cind[i],values[i]);

    free(rind); rind=NULL;
    free(cind); cind=NULL;
    free(values); values=NULL;
/*--------------------------------------------------------------------------*/
/*  same approach but using row compression                                 */
/*--------------------------------------------------------------------------*/

    options[3] = 1;                   /*   row compression => reverse mode, */ 
                                      /* sometimes better than forward mode */ 
                                      /* due to sparsity structure          */

    sparse_jac(tag, m, n, 0, x, &nnz, &rind, &cind, &values, options);

    printf("In sparse format (using row compression): \n");
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

/*  For comparisons: Full Jacobian                                          */

    jacobian(tag,m,n,x,J);

    printmat(" J",m,n,J);
    printf("\n");

/*  repeated call of sparse_jac with same sparsity pattern => repeat = 1 */

    sparse_jac(tag, m, n, 1, x, &nnz, &rind, &cind, &values, options);

    printf("In sparse format:\n");
    for (i=0;i<nnz;i++)
        printf("%2d %2d %10.6f\n\n",rind[i],cind[i],values[i]);

    free(rind); rind=NULL;
    free(cind); cind=NULL;
    free(values); values=NULL;
/*--------------------------------------------------------------------------*/
/*  same approach but using row compression                                 */
/*--------------------------------------------------------------------------*/

    options[3] = 1;                   /*   row compression => reverse mode, */ 
                                      /* sometimes better than forward mode */ 
                                      /* due to sparsity structure          */

    sparse_jac(tag, m, n, 0, x, &nnz, &rind, &cind, &values, options);

    printf("In sparse format (using row compression): \n");
    for (i=0;i<nnz;i++)
        printf("%2d %2d %10.6f\n\n",rind[i],cind[i],values[i]);

    free(rind); rind=NULL;
    free(cind); cind=NULL;
    free(values); values=NULL;
/****************************************************************************/
/*******       sparse Jacobians, separate drivers             ***************/
/****************************************************************************/

/*--------------------------------------------------------------------------*/
/*                                                sparsity pattern Jacobian */
/*--------------------------------------------------------------------------*/

    unsigned int  **JP=NULL;                /* compressed block row storage */
    int ctrl[3];

    JP = (unsigned int **) malloc(m*sizeof(unsigned int*));
    ctrl[0] = 0;
    ctrl[1] = 0;
    ctrl[2] = 0;

    jac_pat(tag, m, n, x, JP, ctrl);

    printf("\n");
    printf("Sparsity pattern of Jacobian: \n");
    for (i=0;i<m;i++) {
        printf(" %d: ",i);
        for (j=1;j<= (int) JP[i][0];j++)
            printf(" %d ",JP[i][j]);
        printf("\n");
    }
    printf("\n");


/*--------------------------------------------------------------------------*/
/*                                                              seed matrix */
/*--------------------------------------------------------------------------*/

    double **Seed;
    int p;
    int option = 0;

    /* option = 0 column compression (default), 
       option = 1 rom compression                */
 
    generate_seed_jac(m, n, JP, &Seed, &p, option);

    printf(" p_J = %d \n",p);
    printmat(" Seed matrix",n,p,Seed);
    printf("\n");

/*--------------------------------------------------------------------------*/
/*                                                      compressed Jacobian */
/*--------------------------------------------------------------------------*/

    double **Jcomp;
    Jcomp = myalloc2(m,p);

    fov_forward(tag,m,n,p,x,Seed,c,Jcomp);
    printmat("compressed J:",m,p,Jcomp);
    printf("\n");


/*--------------------------------------------------------------------------*/
/*  change value of x, but not the sparsity pattern                         */
/*--------------------------------------------------------------------------*/

    for(i=0;i<n;i++)
        x[i] = 2.0*i;

/*  For comparisons: Full Jacobian                                          */

    jacobian(tag,m,n,x,J);

    printmat(" J",m,n,J);
    printf("\n");


    fov_forward(tag,m,n,p,x,Seed,c,Jcomp);
    printmat("compressed J:",m,p,Jcomp);
    printf("\n");

    for (i=0;i<m;i++)
	free(JP[i]);
    free(JP);
    myfree2(J);

    for (i = 0; i < n; i++)
        delete[] Seed[i];
    delete[] Seed;

    myfree2(Jcomp);
}


/***************************************************************************/

void ceval(double *x, double *c) {
    c[0] = 2*x[0]+x[1]-2.0;
    c[1] = x[2]*x[2]+x[3]*x[3]-2.0;
    c[2] = 3*x[4]*x[5] - 3.0;
}

/***************************************************************************/

void ceval_ad(adouble *x, adouble *c) {
    c[0] = 2*x[0]+x[1]-2.0;
    c[0] += cos(x[3])*sin(x[4]);
    c[1] = x[2]*x[2]+x[3]*x[3]-2.0;
    c[2] = 3*x[4]*x[5] - 3.0+sin(x[4]*x[5]);
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
