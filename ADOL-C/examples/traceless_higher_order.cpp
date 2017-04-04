/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     traceless_higher_order.cpp
 Revision: $Id: traceless_higher_order.cpp ??? $
 Contents: computation of partial derivatives
           traceless forward mode for higher order derivatives
           described in the manual

 Copyright (c) Andrea Walther, Andreas Kowarz, Benjamin Jurgelucks
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/


/****************************************************************************/
/*                                                                 INCLUDES */

#define ADOLC_TRACELESS_HIGHER_ORDER
//#include <adolc/adouble.h>
//typedef adhotl::adouble adouble;
#include <iostream>
#include <complex>


#include <adolc/adtl_hov.h>
typedef adtl_hov::adouble adouble;




using namespace std;

adouble poly (adouble* z){
  adouble y;
  y =exp(z[0])*log(pow(z[0],2.0));
  return y;
}

double poly (double* z){
  double y;
  y = exp(z[0])*log(pow(z[0],2.0));
  return y;
}

adouble higher_order(adouble* z)
{
  double a=1.0;
  adouble y;
  y=z[0]*exp(a*z[0]);
  return y;
}

double higher_order(double* z)
{
  double a=1.0;
  double y; 
  y=z[0]*exp(a*z[0]);
  return y;
}

double higher_order_analytic(double* z, int n)
{
  double a=1.0;
  double y; 
  y=exp(a*z[0])*(n*pow(a,n-1)+pow(a,n)*z[0]);
  return y;
}

int factorial(int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}


int main(){
  int d=12, n=3;
  cout << "Degree of derivative (precise up to degree 12):  ";
  cin >> d;
  cout << endl;
  
  double one[1];
  one[0]=1;
  
  //number of Taylor coefficients y_i
  adtl_hov::setDegree(d);
  
  adouble* z = new adouble[n];
  adouble y;
  double* dz = new double[n];
  double dy;
     
  //initialization of independent variables
  for(int i=0; i<n; i++)
  {
    z[i]=2.0;
    dz[i]=2.0;
  }
      
  //set Taylor coefficients x_i
  z[0].setOneADValue(0,one);
  
  double* ret;
  if(d==2)  // analytical derivative only for d=2 available
  {
      //function evaluation for case 1
      y=poly(z);
      dy= poly(dz);
      
      ret=y.getOneADValue(d-1);
      cout << "factorial(d):  " << factorial(d) << endl;
      cout << "ADOLC:    Value of primal: "<< y.getValue() <<"    Value of derivative: " <<  factorial(d)*ret[0] << endl;
      cout << "DOUBLE:   Value of primal: "<< dy <<"    Value of derivative: " <<  (exp(dz[0]) *(-2 + 4*dz[0] + dz[0]*dz[0]*log(dz[0]*dz[0])))/ (dz[0]*dz[0])  << endl;
  }
 
   //function evaluation for case 2
   y=higher_order(z);
   dy=higher_order(dz);
  
   ret=y.getOneADValue(d-1);

  cout << "ADOLC:    Value of primal: "<< y.getValue() <<"    Value of derivative: " << factorial(d)*ret[0] << endl;
  cout << "DOUBLE:   Value of primal: "<< dy <<"    Value of derivative: " <<  higher_order_analytic(dz,d)  << endl;
 
 
  delete []dz;
  delete []z;
}

