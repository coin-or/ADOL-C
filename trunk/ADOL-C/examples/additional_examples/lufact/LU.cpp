/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     LU.cpp
 Revision: $Id$
 Contents: example for 'active' LU-decomposition and according solver

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#define _LU_C

/****************************************************************************/
/*                                                                 INCLUDES */
#include "LU.h"

/****************************************************************************/
/*                                                          ADOUBLE ROUTINE */
/* Simple LU-factorization according to Crout's algorithm without pivoting */
void LUfact(int n, adouble **A) {
    int i, j, k;
    adouble dum;
    for (j=0; j<n; j++) { /* L-part */
        for (i=0; i<j; i++)
            for (k=0; k<i; k++)
                A[i][j] -= A[i][k] * A[k][j];
        /* U-part */
        for (i=j; i<n; i++)
            for (k=0; k<j; k++)
                A[i][j] -= A[i][k] * A[k][j];
        if (A[j][j] != 0) {
            dum = 1.0 / A[j][j];
            for (i=j+1; i<n; i++)
                A[i][j] *= dum;
        } else {
            fprintf(stderr,"Error in LUfact(..): pivot is zero\n");
            exit(-99);
        }
    }
}


/****************************************************************************/
/*                                                          ADOUBLE ROUTINE */
/* Solution of A*x=b by forward and backward substitution */
void LUsolve(int n, adouble **A, adouble *bx) {
    int i, j;
    /* forward substitution */
    for (i=0; i<n; i++)
        for (j=0; j<i-1; j++)
            bx[i] -= A[i][j] * bx[j];
    /* backward substitution */
    for (i=n-1; i>=0; i--) {
        for (j=i+1; j<n; j++)
            bx[i] -= A[i][j] * bx[j];
        bx[i] /= A[i][i];
    }
}


/****************************************************************************/
/*                                                              END OF FILE */
#undef _LU_C
