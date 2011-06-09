/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fortutils.c
 Revision: $Id$
 Contents: Internal tools to handle Fortran arrays
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#include <adolc/fortutils.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                                          */

/*--------------------------------------------------------------------------*/
void spread1(int m, fdouble* x, double* X) {
    int j;
    for (j=0; j<m; j++)
        X[j] = *x++;
}

/*--------------------------------------------------------------------------*/
void pack1(int m, double* X, fdouble* x) {
    int j;
    for (j=0; j<m; j++)
        *x++ = X[j];
}

/*--------------------------------------------------------------------------*/
void spread2(int m, int n, fdouble* x, double** X) {
    int i,j;
    for (j=0; j<n; j++)
        for (i=0; i<m; i++)
            X[i][j] = *x++;
}

/*--------------------------------------------------------------------------*/
void pack2(int m, int n, double** X, fdouble* x) {
    int i,j;
    for (j=0; j<n; j++)
        for (i=0; i<m; i++)
            *x++ = X[i][j];
}

/*--------------------------------------------------------------------------*/
void spread3(int m, int n, int p, fdouble* x, double*** X) {
    int i,j,k;
    for (k=0; k<p; k++)
        for (j=0; j<n; j++)
            for (i=0; i<m; i++)
                X[i][j][k] = *x++;
}

/*--------------------------------------------------------------------------*/
void pack3(int m, int n, int p, double*** X, fdouble* x) {
    int i,j,k;
    for (k=0; k<p; k++)
        for (j=0; j<n; j++)
            for (i=0; i<m; i++)
                *x++ = X[i][j][k];
}

/****************************************************************************/
END_C_DECLS
