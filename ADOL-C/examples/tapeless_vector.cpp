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

#include <adolc/adtl.h>
typedef adtl::adouble adouble;

int main(int argc, char *argv[]) {
    adtl::setNumDir(3);
    adouble x[3], y[3];

    for (int i=0; i<3;++i)          // Initialize x_i
    {
        x[i] = i + 1.0;
        for (int j=0; j<3;++j)
            if (i==j)
                x[i].setADValue(j,1);
    }

    cout << endl << "Transform from Cartesian to spherical polar coordinates" << endl << endl;

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
    for (int i=0; i<3;++i) {
        for (int j=0; j<3;++j)
            cout << y[i].getADValue(j) << "  ";
        cout << endl;
    }
    cout << endl;

    return 0;
}

