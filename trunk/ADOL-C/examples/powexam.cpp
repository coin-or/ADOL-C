/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     powexam.cpp
 Revision: $Id$
 Contents: computation of n-th power, described in the manual

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>             /* use of ALL ADOL-C interfaces */

#include <iostream>
using namespace std;

/****************************************************************************/
/*                                                          ADOUBLE ROUTINE */
adouble power(adouble x, int n) {
    adouble z = 1;

    if (n>0)                           /* Recursion and branches */
    { int nh = n/2;                    /* that do not depend on  */
        z = power(x,nh);                 /* adoubles are fine !!!! */
        z *= z;
        if (2*nh != n)
            z *= x;
        return z;
    } /* end if */
    else {
        if (n==0)                        /* The local adouble z dies */
            return z;                      /* as it goes out of scope. */
        else
            return 1/power(x,-n);
    } /* end else */
} /* end power */

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
    int i,tag = 1;
    int n;

    cout << "COMPUTATION OF N-TH POWER (ADOL-C Documented Example)\n\n";
    cout << "monomial degree=? \n";    /* input the desired degree */
    cin >> n;
    /* allocations and initializations */
    double** X;
    double** Y;
    X = myalloc2(1,n+4);
    Y = myalloc2(1,n+4);
    X[0][0] = 0.5;                   /* function value = 0. coefficient */
    X[0][1] = 1.0;                   /* first derivative = 1. coefficient */
    for(i=0; i<n+2; i++)
        X[0][i+2] = 0;                 /* further coefficients */
    double** Z;                      /* used for checking consistency */
    Z = myalloc2(1,n+2);             /* between forward and reverse */

    adouble y,x;                     /* declare active variables */
    /* beginning of active section */
    trace_on(tag);                   /* tag = 1 and keep = 0 */
    x <<= X[0][0];                 /* only one independent var */
    y = power(x,n);                /* actual function call */
    y >>= Y[0][0];                 /* only one dependent adouble */
    trace_off();                     /* no global adouble has died */
    /* end of active section */
    double u[1];                     /* weighting vector */
    u[0]=1;                          /* for reverse call */
    for(i=0; i<n+2; i++)             /* note that keep = i+1 in call */
    { forward(tag,1,1,i,i+1,X,Y);    /* evaluate the i-the derivative */
        if (i==0)
            cout << Y[0][i] << " - " << y.value() << " = " << Y[0][i]-y.value()
            << " (should be 0)\n";
        else {
            Z[0][i] = Z[0][i-1]/i;       /* scale derivative to Taylorcoeff. */
            cout << Y[0][i] << " - " << Z[0][i] << " = " << Y[0][i]-Z[0][i]
            << " (should be 0)\n";
        }
        reverse(tag,1,1,i,u,Z);        /* evaluate the (i+1)-st deriv. */
    } /* end for */

    return 1;
} /* end main */

