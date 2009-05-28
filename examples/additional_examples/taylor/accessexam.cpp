/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     accessexam.cpp
 Revision: $Id$
 Contents: Test driver 'tensor_value(..)' and access of higher order tensors 
           by void pointer strategy
 
 Copyright (c) Andrea Walther, Andreas Griewank
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <stdlib.h>
#include <iostream>
using namespace std;

/****************************************************************************/
/*                                                                     MAIN */
int main() {
    int i,j,n,m,d,p,dim;

    /*--------------------------------------------------------------------------*/
    cout << "ACCESSEXAM (ADOL-C Example)\n\n";                      /* inputs */
    cout << " demonstrates accees to higher order "
    << "partial derivatives (tensors)\n\n";
    cout << " Number of independents ?\n ";
    cin >> n;
    cout << " Number of dependents (2 <= m <=n) ?\n ";
    cin >> m;
    d = 5;
    cout << " Degree = " << d <<"\n";
    cout << " Number of directions ?\n ";
    cin >> p;

    /*--------------------------------------------------------------------------*/
    int* multi = new int[d];                         /* allocations and inits */
    double* xp = new double[n];
    double* yp = new double[m];
    double** S = new double*[n];
    double* test = new double[m];
    double** tensorhelp;
    double****** tensor;

    for (i=0; i<n; i++) {
        xp[i] = (i+1.0)/(2.0+i);
        S[i] = new double[p];
        for (j=0; j<p; j++)
            S[i][j]=(i==j)?1.0:0.0;
    }

    for (i=0; i<d; i++)
        multi[i] = 0;

    /*--------------------------------------------------------------------------*/
    trace_on(1);                                      /* tracing the function */
    adouble* x = new adouble[n];
    adouble* y = new adouble[m];
    y[0] = 1;

    for (i=0; i<n; i++) {
        x[i] <<= xp[i];
        y[0] *= x[i];
    }
    y[1] = sqrt(x[1]);
    for (i=2; i<m; i++)
        y[i] = sqrt(x[i]);
    for (i=0; i<m; i++)
        y[i] >>= yp[i] ;
    trace_off();

    /*--------------------------------------------------------------------------*/
    dim = binomi(p+d,d);                                       /* tensor_eval */
    tensorhelp = myalloc2(n,dim);
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";
    tensor_eval(1,m,n,d,p,xp,tensorhelp,S);

    /*--------------------------------------------------------------------------*/
    tensor = (double******) tensorsetup(n,p,d,tensorhelp);          /* access */
    cout<<"\nTASK 1: Check access to directional derivatives\n";

    for (i=0; i<p; i++) {
        multi[0] = i+1;
        tensor_value(d,m,test,tensorhelp,multi);
        cout << i+1 << ": ";
        for (j=0; j<m; j++)
            cout << " " << test[j] << " ";
        cout << "==";
        for (j=0; j<m; j++)
            cout << " " << tensor[j][i+1][0][0][0][0] << " ";
        cout << "\n";
    }

    /*--------------------------------------------------------------------------*/
    cout<<"\nTASK 2: Check access to higher order derivatives\n";
    multi[0] = 2;
    multi[1] = 2;
    cout <<" Tensor";
    for (i=0; i<d; i++)
        cout << "[" << multi[i] << "]";
    cout << "\n";
    tensor_value(d,m,test,tensorhelp,multi);
    cout << " " << test[1] << " == ";
    cout << tensor[1][2][2][0][0][0] << "\n\n";

    /*--------------------------------------------------------------------------*/
    multi[0] = 2;
    multi[1] = 2;
    multi[2] = 2;
    cout <<" Tensor";
    for (i=0; i<d; i++)
        cout << "[" << multi[i] << "]";
    cout << "\n";
    tensor_value(d,m,test,tensorhelp,multi);
    cout << " " << test[1] << " == ";
    cout << tensor[1][2][2][2][0][0] << "\n\n";

    /*--------------------------------------------------------------------------*/
    multi[0] = 2;
    multi[1] = 2;
    multi[2] = 2;
    multi[3] = 2;
    cout <<" Tensor";
    for (i=0; i<d; i++)
        cout << "[" << multi[i] << "]";
    cout << "\n";
    tensor_value(d,m,test,tensorhelp,multi);
    cout << " " << test[1] << " == ";
    cout << tensor[1][2][2][2][2][0] << "\n\n";

    /*--------------------------------------------------------------------------*/
    multi[0] = 2;
    multi[1] = 2;
    multi[2] = 2;
    multi[3] = 2;
    multi[4] = 2;
    cout<<" Tensor";
    for (i=0; i<d; i++)
        cout << "[" << multi[i] << "]";
    cout << "\n";
    tensor_value(d,m,test,tensorhelp,multi);
    cout << " " << test[1] << " == ";
    cout << tensor[1][2][2][2][2][2] << "\n\n";
    freetensor(n,p,d, (double **) tensor);

    return 1;
}

/****************************************************************************/
/*                                                               THAT'S ALL */
