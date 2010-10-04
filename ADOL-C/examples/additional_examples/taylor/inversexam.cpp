/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     inversexam.cpp
 Revision: $Id$
 Contents: Test driver 'inverse_tensor_eval(..)' allows to 
           compute higher order derivatives of inverse
           functions

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

    /*--------------------------------------------------------------------------*/
    cout << "INVERSEXAM (ADOL-C Example)\n\n";                      /* inputs */
    cout << " Number of independents = ?\n ";
    cin >> n;
    // number of dependents = number of independents !!
    cout << " Degree = ?\n ";
    cin >> d;
    cout << " Number of directions = ?\n ";
    cin >> p;

    /*--------------------------------------------------------------------------*/
    int* multi = new int[d];                         /* allocations and inits */
    double* xp = new double[n];
    double* yp = new double[n];
    double** S = new double*[n];
    double* test = new double[n];
    double** tensoren;
    adouble* x = new adouble[n];
    adouble* y = new adouble[n];

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
    for (i=0; i<n; i++) {
        x[i] <<= xp[i];
        y[i] = (i+1)*x[i];
    }
    y[0] += sqrt(x[0]);
    for (i=0; i<n; i++)
        y[i] >>= yp[i] ;
    trace_off();

    /*--------------------------------------------------------------------------*/
    d = d-1;                                        /* 1. inverse_tensor_eval */
    dim = binomi(p+d,d);
    tensoren = myalloc2(n,dim);
    cout <<"TASK 1:\n";
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";

    inverse_tensor_eval(1,n,d,p,xp,tensoren,S);

    for (i=0; i<p; i++) {
        multi[0] = i+1;
        tensor_value(d,n,test,tensoren,multi);
        cout << i+1 << ": ";
        for (j=0; j<n; j++)
            cout << " " << test[j] << " ";
        cout << "\n";
    }

    myfree2(tensoren);

    /*--------------------------------------------------------------------------*/
    d = d+1;                                        /* 2. inverse_tensor_eval */
    dim = binomi(p+d,d);
    cout <<"TASK 2:\n";
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";
    tensoren = myalloc2(n,dim);

    inverse_tensor_eval(1,n,d,p,xp,tensoren,S);

    for (i=0; i<p; i++) {
        multi[0] = i+1;
        tensor_value(d,n,test,tensoren,multi);
        cout << i+1 << ": ";
        for (j=0; j<n; j++)
            cout << " " << test[j] << " ";
        cout << "\n";
    }

    /*--------------------------------------------------------------------------*/
    xp[0] = 2*xp[0];                                /* 3. inverse_tensor_eval */

    cout <<"TASK 3:\n";
    cout <<" NEW independend values !!!\n";
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";

    inverse_tensor_eval(1,n,d,p,xp,tensoren,S);

    for (i=0; i<p; i++) {
        multi[0] = i+1;
        tensor_value(d,n,test,tensoren,multi);
        cout << i+1 << ": ";
        for (j=0; j<n; j++)
            cout << " " << test[j] << " ";
        cout << "\n";
    }

    myfree2(tensoren);

    /*--------------------------------------------------------------------------*/
    d = d-1;                                        /* 4. inverse_tensor_eval */
    dim = binomi(p+d,d);
    cout <<"TASK 4:\n";
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";
    tensoren = myalloc2(n,dim);

    inverse_tensor_eval(1,n,d,p,xp,tensoren,S);

    for (i=0; i<p; i++) {
        multi[0] = i+1;
        tensor_value(d,n,test,tensoren,multi);
        cout << i+1 << ": ";
        for (j=0; j<n; j++)
            cout << " " << test[j] << " ";
        cout << "\n";
    }

    myfree2(tensoren);

    /*--------------------------------------------------------------------------*/
    d = d+1;                                        /* 5. inverse_tensor_eval */
    dim = binomi(p+d,d);
    xp[0] = 0.5*xp[0];
    cout <<"TASK 5:\n";
    cout <<" OLD independend values !!!\n";
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";
    tensoren = myalloc2(n,dim);

    inverse_tensor_eval(1,n,d,p,xp,tensoren,S);

    for (i=0; i<p; i++) {
        multi[0] = i+1;
        tensor_value(d,n,test,tensoren,multi);
        cout << i+1 << ": ";
        for (j=0; j<n; j++)
            cout << " " << test[j] << " ";
        cout << "\n";
    }

    myfree2(tensoren);

    return 1;
}


/****************************************************************************/
/*                                                               THAT'S ALL */
