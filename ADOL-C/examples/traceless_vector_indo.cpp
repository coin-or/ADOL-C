/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     tapeless_vector.cpp
 Revision: $Id$
 Contents: computation of coordinate transform, 
           vector tapeless forward mode
           described in the manual

 Copyright (c) Andrea Walther, Andreas Kowarz
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */

#include <iostream>
using namespace std;

#include <adolc/adtl_indo.h>
#include <adolc/sparse/sparsedrivers.h>

template<typename T>
class my_function : public func_ad<T> {
public:
    int operator() (int n, T *x, int m, T *y)
    {
        y[0] = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        y[1] = atan(sqrt(x[0]*x[0]+x[1]*x[1])/x[2]);
        y[2] = atan(x[1]/x[0]);
        return 1;
    }
};


int main(int argc, char *argv[]) {
    const int m=3, n=3;
    adtl::setNumDir(n);

    my_function<adtl::adouble> fun;
    my_function<adtl_indo::adouble> fun_indo;
    adtl::adouble x[n], y[m];

    for (int i=0; i<n;++i)          // Initialize x_i
    {
        x[i] = i + 1.0;
        for (int j=0; j<m;++j)
            if (i==j)
                x[i].setADValue(j,1);
    }

    cout.precision(15);
    cout << endl << "Transform from Cartesian to spherical polar coordinates" << endl << endl;

    cout << "cartesian coordinates: " << endl;
    cout << "x[0] = " << x[0].getValue() << "  x[1] = " << x[1].getValue()
    << "  x[2] = " << x[2].getValue() << endl << endl;

    fun(3, x, 3, y);

    cout << "cpherical polar coordinates: " << endl;
    cout << "y[0] = " << y[0].getValue() << "  y[1] = " << y[1].getValue()
    << "  y[2] = " << y[2].getValue() << endl <<endl;

    // "use" the derivative
    cout << "derivatives:" << endl;
    for (int i=0; i<3;++i) {
        for (int j=0; j<3;++j)
            cout << y[i].getADValue(j) << "  ";
        cout << endl;
    }
    cout << endl;

    double* basepoints = new double[n];
    for(int i = 0; i<n; i++)
        basepoints[i] = x[i].getValue();
    int retVal;
    int nnz;
    unsigned int *rind = NULL, *cind = NULL;
    double *values = NULL;
    retVal = ::ADOLC_get_sparse_jacobian(&fun, &fun_indo, n, m, 0, basepoints, &nnz, &rind, &cind, &values);

    cout << endl;
    cout << "Checking results with ADOLC_get_sparse_jacobian functionality..." << endl;
    cout << "number of non-zero elements in jacobian: " << nnz << endl;
    cout << "Elements are:" << endl;

    for(int i = 0; i < nnz; i++)
    {
      cout << "[" << *rind<< "][" << *cind << "]: " << *values << endl;
      rind++;
      cind++;
      values++;
    }
    
    return 0;
}

