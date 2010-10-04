/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     coordinates.cpp
 Revision: $Id$
 Contents: Test driver 'inverse_tensor_eval' using transformation between 
           Cartesian coordinates and polar coordinates
 
 Copyright (c) Andrea Walther, Andreas Griewank

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <cstdlib>
#include <iostream>
using namespace std;

/****************************************************************************/
/*                                                                     MAIN */
int main() {
    int i,j,n,d,p,dim;
    double zp[4];
    double gp[2];
    double zd[2];

    /*--------------------------------------------------------------------------*/
    cout << "COORDINATES (ADOL-C Example)\n\n";                     /* inputs */
    cout << " Cartesian coordinates:\n";
    cout << " z_1: (e.g. 4) \n";
    cin >> zp[0];
    cout << " z_2: (e.g. 3) \n";
    cin >> zp[1];
    cout << "\n Polar coordinates:\n";
    cout << " z_3: (e.g. 5) \n";
    cin >> zp[2];
    cout << " z_4: (e.g. 0.64350110879) \n";
    cin >> zp[3];

    cout << "\n Highest derivative degree = 3\n";

    /*--------------------------------------------------------------------------*/
    /* allocations and inits */
    n = 4;
    p = 2;
    d = 3;

    double** S = new double*[n];
    double** tensor = new double*[n];
    double**** tensorentry;

    for (i=0; i<n; i++) {
        S[i] = new double[p];
        for(j=0;j<p;j++)
            S[i][j]=0.0;
    }
    S[2][0] = 1;
    S[3][1] = 1;

    /*--------------------------------------------------------------------------*/
    trace_on(1);                                      /* tracing the function */
    adouble* z = new adouble[n];
    adouble* g = new adouble[2];

    for(i=0;i<n;i++)
        z[i] <<= zp[i];
    g[0] = z[0]*z[0] + z[1]*z[1] - z[2]*z[2];
    g[1] = cos(z[3]) - z[0]/z[2];
    g[0] >>= gp[0];
    g[1] >>= gp[1];
    z[0] >>= zd[0];
    z[2] >>= zd[1];
    trace_off();

    /*--------------------------------------------------------------------------*/
    dim = binomi(p+d,d);                               /* inverse_tensor_eval */
    for(i=0;i<n;i++) {
        tensor[i] = new double[dim];
        for(j=0;j<dim;j++)
            tensor[i][j] = 0;
    }
    inverse_tensor_eval(1,n,d,p,zp,tensor,S);
    tensorentry = (double****) tensorsetup(n,p,d,tensor);
    cout<< "\n Some partial derivatives: \n\n";
    cout <<"     Tensor";
    cout << "[1][1][0][0]: \t";
    cout << tensorentry[1][1][0][0] << "\n\n";
    cout <<"     Tensor";
    cout << "[1][2][0][0]: \t";
    cout << tensorentry[1][2][0][0] << "\n\n";
    cout <<"     Tensor";
    cout << "[1][2][1][0]: \t";
    cout << tensorentry[1][2][1][0] << "\n\n";
    free((char*) *tensor);
    free((char*)tensor);
    freetensor(n,p,d, (double **) tensorentry);

    return 1;
}

/****************************************************************************/
/*                                                               THAT'S ALL */
