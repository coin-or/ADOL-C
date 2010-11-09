/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     taylorexam.cpp
 Revision: $Id$
 Contents: Test driver 'tensor_eval(..)' to compute
           higher order derivatives

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
    int i,j,m,n,d,p,dim;

    /*--------------------------------------------------------------------------*/
    cout << "TAYLOREXAM (ADOL-C Example)\n\n";                      /* inputs */
    cout << " Number of indenpendents = ?\n ";
    cin >> n;
    cout << " Number of dependents = (<=n) ?\n ";
    cin >> m;
    cout << " Degree = ?\n ";
    cin >> d;
    cout << " Number of directions = ?\n ";
    cin >> p;

    /*--------------------------------------------------------------------------*/
    int* multi = new int[d];                         /* allocations and inits */
    double* xp = new double[n];
    double* yp = new double[m];
    double** S = new double*[n];
    double* test = new double[m];
    double** tensoren;
    adouble* x = new adouble[n];
    adouble* y = new adouble[m];

    for (i=0; i<d; i++)
        multi[i] = 0;

    for (i=0; i<n; i++) {
        xp[i] = (i+1.0)/(2.0+i);
        S[i] = new double[p];
        for (j=0; j<p; j++)
            S[i][j] = (i==j)?1.0:0.0;
    }

    /*--------------------------------------------------------------------------*/
    trace_on(1);                                       /* tracing the function */
    // adouble* x = new adouble[n];
    // adouble* y = new adouble[m];
    y[0] = 1;

    for (i=0; i<n; i++) {
        x[i] <<= xp[i];
        y[0] *= x[i];
    }
    for (i=1; i<m; i++)
        y[i] = x[i];
    for (i=0; i<m; i++)
        y[i] >>= yp[i] ;
    trace_off();

    /*--------------------------------------------------------------------------*/
    d = d-1;                                                /* 1. tensor_eval */
    dim = binomi(p+d,d);
    cout <<"TASK 1:\n";
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";
    tensoren = myalloc2(m,dim);

    tensor_eval(1,m,n,d,p,xp,tensoren,S);

    for (i=0; i<p; i++) {
        multi[0] = i+1;
        tensor_value(d,m,test,tensoren,multi);
        cout << i+1 << ": ";
        for(j=0; j<m; j++)
            cout << " " << test[j] << " ";
        cout << "\n";
    }

    myfree2(tensoren);

    /*--------------------------------------------------------------------------*/
    d = d+1;                                                /* 2. tensor_eval */
    dim = binomi(p+d,d);
    cout <<"TASK 2:\n";
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";
    tensoren = myalloc2(m,dim);

    tensor_eval(1,m,n,d,p,xp,tensoren,S);

    for(i=0; i<p; i++) {
        multi[0] = i+1;
        tensor_value(d,m,test,tensoren,multi);
        cout << i+1 << ": ";
        for (j=0; j<m; j++)
            cout << " " << test[j] << " ";
        cout << "\n";
    }
    cout << "\n";


    /*--------------------------------------------------------------------------*/
    xp[0] = 2*xp[0];                                        /* 3. tensor_eval */

    cout <<"TASK 3:\n";
    cout <<" NEW independend values !!!\n";
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";

    tensor_eval(1,m,n,d,p,xp,tensoren,S);

    for(i=0; i<p; i++) {
        multi[0] = i+1;
        tensor_value(d,m,test,tensoren,multi);
        cout << i+1 << ": ";
        for (j=0; j<m; j++)
            cout << " " << test[j] << " ";
        cout << "\n";
    }

    myfree2(tensoren);

    /*--------------------------------------------------------------------------*/
    d = d-1;                                                /* 4. tensor_eval */
    dim = binomi(p+d,d);
    cout <<"TASK 4:\n";
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";
    tensoren = myalloc2(m,dim);

    tensor_eval(1,m,n,d,p,xp,tensoren,S);
    for(i=0;i<p;i++) {
        multi[0] = i+1;
        tensor_value(d,m,test,tensoren,multi);
        cout << i+1 << ": ";
        for (j=0; j<m; j++)
            cout << " " << test[j] << " ";
        cout << "\n";
    }

    myfree2(tensoren);

    /*--------------------------------------------------------------------------*/
    d = d+1;                                                /* 5. tensor_eval */
    dim = binomi(p+d,d);
    xp[0] = 0.5*xp[0];
    cout <<"TASK 5:\n";
    cout <<" OLD independend values !!!\n";
    cout <<" d = "<<d<<", dim = "<<dim<<"\n";
    tensoren = myalloc2(m,dim);

    tensor_eval(1,m,n,d,p,xp,tensoren,S);
    for(i=0;i<p;i++) {
        multi[0] = i+1;
        tensor_value(d,m,test,tensoren,multi);
        cout << i+1 << ": ";
        for (j=0; j<m; j++)
            cout << " " << test[j] << " ";
        cout << "\n";
    }

    myfree2(tensoren);

    return 1;
}


/****************************************************************************/
/*                                                               THAT'S ALL */

