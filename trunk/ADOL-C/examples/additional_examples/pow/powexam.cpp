/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     powexam.cpp
 Revision: $Id$
 Contents: example for computation of n-th power

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */

#include <adolc/adolc.h>               // use of ALL ADOL-C interfaces

#include <iostream>
using namespace std;

#include <math.h>

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
    int i,tag = 1;
    int n;
    double e;

    cout << "COMPUTATION OF x^e\n\n";
    cout << "e=? \n";    // input the desired degree
    cin >> e;

    n = (int) e;
    if (n < 0)
        n = -n;
    // allocations and initializations
    double** X;
    double** Y;
    X = myalloc2(1,n+4);
    Y = myalloc2(1,n+4);
    cout << "value of x=? \n";
    cin >> X[0][0];                  // function value = 0. coefficient
    X[0][1] = 1.0;                   // first derivative = 1. coefficient
    X[0][2] = 0.0;                   // second derivative = 2. coefficient
    for(i=1; i<n+2; i++)
        X[0][i+2] = 0;                 // further coefficients
    double** Z;                      // used for checking consistency
    Z = myalloc2(1,n+2);             // between forward and reverse

    adouble y,x;                     // declare active variables
    // beginning of active section
    trace_on(tag);                   // tag = 1 and keep = 0
    x <<= X[0][0];                 // only one independent var
    y = 1;
    for(i=0;i<n;i++)
        y *= x;                        // actual function call
    y >>= Y[0][0];                 // only one dependent adouble
    trace_off();                     // no global adouble has died
    // end of active section */
    double u[1];                     // weighting vector
    u[0]=1;                          // for reverse call
    for(i=0; i<n+2; i++)             // note that keep = i+1 in call
    { forward(tag,1,1,i,i+1,X,Y);    // evaluate the i-the derivative
        cout << "Result: " << Y[0][0] << "\n";
        if (i==0)
            cout << i << " " << Y[0][i] << " - " << y.value() << " = " << Y[0][i]-y.value()
            << " (should be 0)\n";
        else {
            Z[0][i] = Z[0][i-1]/i;       // scale derivative to Taylorcoeff.
            cout << i << " " << Y[0][i] << " - " << Z[0][i] << " = " << Y[0][i]-Z[0][i]
            << " (should be 0)\n";
        }
        reverse(tag,1,1,i,u,Z);        // evaluate the (i+1)-st deriv.
    } // end for

    cout << "\n\n";
    trace_on(tag);                   // tag = 1 and keep = 0
    x <<= X[0][0];                 // only one independent var
    y = pow(x,e);                  // actual function call
    y >>= Y[0][0];                 // only one dependent adouble
    trace_off();                     // no global adouble has died
    // end of active section */
    u[0]=1;                          // for reverse call
    for(i=0; i<n+2; i++)             // note that keep = i+1 in call
    { forward(tag,1,1,i,i+1,X,Y);    // evaluate the i-the derivative
        cout << "Result: " << Y[0][0] << "\n";
        if (i==0)
            cout << i << " " << Y[0][i] << " - " << y.value() << " = " << Y[0][i]-y.value()
            << " (should be 0)\n";
        else {
            Z[0][i] = Z[0][i-1]/i;       // scale derivative to Taylorcoeff.
            cout << i << " " << Y[0][i] << " - " << Z[0][i] << " = " << Y[0][i]-Z[0][i]
            << " (should be 0)\n\n";
        }
        reverse(tag,1,1,i,u,Z);        // evaluate the (i+1)-st deriv.
    } // end for

    return 1;
} // end main

