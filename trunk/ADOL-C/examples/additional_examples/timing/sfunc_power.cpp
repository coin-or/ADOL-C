/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sfunc_power.cpp
 Revision: $Id$
 Contents: function module containing the power example

   Each << function module >> contains:
          
     (1) const char* const controlFileName 
     (2) int indepDim; 
     (3) void initProblemParameters( void )
     (4) void initIndependents( double* indeps )
     (5) double originalScalarFunction( double* indeps )
     (6) double tapingScalarFunction( int tag, double* indeps )   
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#define _SFUNC_POWER_C_

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <cstdlib>
#include <time.h>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "powexam.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */
int exponent;
//static unsigned short int dx[3]; /* variable needed by erand48(.) */


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"COMPUTATION OF n-th POWER (ADOL-C Example)\n\n");
    indepDim = 1;

    fprintf(stdout,"    n = ? ");
    fscanf(stdin,"%d",&exponent);
    fprintf(stdout,"\n");

    /* Init erand48(); */
    struct tm s;
    time_t t;
    time(&t);
    s=*localtime(&t);
    srand(s.tm_sec*s.tm_min);
    /*  dx[0]=rand();
      dx[1]=rand();
      dx[2]=rand();*/
}


/****************************************************************************/
/*                                                        INITIALIZE INDEPs */
void initIndependents( double* indeps ) {
    indeps[0] = (double)rand();
}


/****************************************************************************/
/*                                                 ORIGINAL SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                       The recursive determinant function */
double power( double x, int n ) {
    double z = 1;
    if (n > 0) {
        int nh =n/2;
        z = power(x,nh);
        z *= z;
        if (2*nh != n)
            z *= x;
        return z;
    } else
        if (n == 0)
            return z;
        else
            return 1.0/power(x,-n);
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
double originalScalarFunction( double* indeps ) {
    return power(indeps[0],exponent);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                             The recursive power function */
adouble activePower( adouble x, int n) {
    adouble z = 1;
    if (n > 0) {
        int nh =n/2;
        z = activePower(x,nh);
        z *= z;
        if (2*nh != n)
            z *= x;
        return z;
    } else
        if (n == 0)
            return z;
        else
            return 1.0/activePower(x,-n);
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
double tapingScalarFunction( int tag, double* indeps ) {
    trace_on(tag);
    adouble activeIndep;
    activeIndep <<= *indeps;
    adouble ares = activePower(activeIndep,exponent);
    double res = 0;
    ares >>= res;
    trace_off();
    return res;
}

#undef _SFUNC_POWER_C_





