/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     tapeless_scalar.cpp
 Revision: $Id$
 Contents: computation of coordinate transform, 
           scalar tapeless forward mode
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

#include <adolc/adtl.h>
typedef adtl::adouble adouble;


int main(int argc, char *argv[]) {
    adouble x[3], y[3];

    for (int i=0; i<3;++i)          // Initialize x_i
        x[i] = i + 1.0;

    cout << endl << "Compute transform from Cartesian to spherical polar coordinates" << endl << endl;

    // derivative of y with respect to x0
    double one=1.0;
    x[0].setADValue(&one);

    y[0] = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    y[1] = atan(sqrt(x[0]*x[0]+x[1]*x[1])/x[2]);
    y[2] = atan(x[1]/x[0]);

    cout << "cartesian coordinates: " << endl;
    cout << "x[0] = " << x[0].getValue() << "  x[1] = " << x[1].getValue()
    << "  x[2] = " << x[2].getValue() << endl << endl;
    cout << "cpherical polar coordinates: " << endl;
    cout << "y[0] = " << y[0].getValue() << "  y[1] = " << y[1].getValue()
    << "  y[2] = " << y[2].getValue() << endl <<endl;

    // "use" the derivative
    cout << "derivative:" << endl;
    cout << "dy[0]/dx[0] = " << *y[0].getADValue() << "  dy[1]/dx[0] = " << *y[1].getADValue()
    << "  dy[2]/dx[0] = " << *y[2].getADValue() << endl;

    return 0;
}

