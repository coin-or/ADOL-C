/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     tapeless_higher_order.cpp
 Revision: $Id: tapeless_higher_order.cpp ??? $
 Contents: computation of partial derivatives
           tapeless forward mode for higher order derivatives
           described in the manual

 Copyright (c) Andrea Walther, Andreas Kowarz
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/


/****************************************************************************/
/*                                                                 INCLUDES */

#define ADOLC_TAPELESS_HIGHER_ORDER
#include <adolc/adouble.h>
typedef adhotl::adouble adouble;
#include <iostream>
using namespace std;

adouble poly (adouble* z){
  adouble y;
  y=pow(z[0],3.0)+2.0*pow(z[1],3.0)+3.0*pow(z[2],3.0)+z[0]*z[1]
    +2.0*z[1]*z[2]+3.0*z[0]*z[2]+5.0*z[1]*z[2]*z[2];
  return y;
}

int main(){
  int d=2, n=3;
  
  //number of Taylor coefficients y_i
  adouble::setDegree(d);
  
  adouble* z = new adouble[n];
  adouble y;
 
  //initialization of independent variables
  for(int i=0; i<n; i++)
    z[i]=1.0;
  
  //set Taylor coefficients x_i
  z[2].setOneADValue(0,1.0);
  
  //function evaluation
  y=poly(z);
  
  //partial derivative d^2F/dz3^2
  cout << "Value of derivative: " << 2.0*y.getOneADValue(1) << endl;
  delete []z;
}

