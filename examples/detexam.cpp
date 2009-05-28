/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     detexam.cpp
 Revision: $Id$
 Contents: computation of determinants, described in the manual

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adouble.h>          // use of active doubles
#include <adolc/interfaces.h>       // use of basic forward/reverse
// interfaces of ADOL-C
#include <adolc/taping.h>           // use of taping

#include <iostream>
using namespace std;

/****************************************************************************/
/*                                                          ADOUBLE ROUTINE */
int n;
adouble **A;                        // A is an n x n matrix
adouble zero = 0;

adouble det(int k, int m)           // k <= n is the order of the submatrix
{ if (m == 0)                       // its column indices
        return 1.0;
    else                              // are encoded in m
    {
        adouble *pt = A[k-1];
        adouble   t = zero;
        int p = 1;
        int s;
        if (k%2)
            s = 1;
        else
            s = -1;
        for(int i=0; i<n; i++) {
            int p1 = 2*p;
            if (m%p1 >= p) {
                if (m == p) {
                    if (s>0)
                        t += *pt;
                    else
                        t -= *pt;
                } else {
                    if (s>0)
                        t += *pt*det(k-1, m-p); // recursive call to det
                    else
                        t -= *pt*det(k-1, m-p); // recursive call to det
                }
                s = -s;
            }
            ++pt;
            p = p1;
        }
        return t;
    }
}

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
    int i,j, m = 1;
    int tag = 1;
    int keep = 1;

    cout << "COMPUTATION OF DETERMINANTS (ADOL-C Documented Example)\n\n";
    cout << "order of matrix = ? \n"; // select matrix size
    cin >> n;

    A = new adouble*[n];
    adouble ad;

    trace_on(tag,keep);               // tag=1=keep
    double detout = 0.0, diag = 1.0;// here keep the intermediates for
    for (i=0; i<n; i++)             // the subsequent call to reverse
    { m *= 2;
        A[i] = new adouble[n];
        for (j=0; j<n; j++)
            A[i][j] <<= j/(1.0+i);      // make all elements of A independent
        diag += A[i][i].value();       // value(adouble) converts to double
        A[i][i] += 1.0;
    }
    ad = det(n,m-1);                // actual function call.
    ad >>= detout;
    printf("\n %f - %f = %f  (should be 0)\n",detout,diag,detout-diag);
    trace_off();

    double u[1];
    u[0] = 1.0;
    double* B = new double[n*n];

    reverse(tag,1,n*n,0,u,B);         // call reverse to calculate the gradient

    cout << " \n first base? : ";
    for (i=0; i<n; i++) {
        adouble sum = 0;
        for (j=0; j<n; j++)             // the matrix A times the first n
            sum += A[i][j]*B[j];          // components of the gradient B
        cout << sum.value() << " ";      // must be a Cartesian basis vector
    }
    cout << "\n";

    return 1;
}

