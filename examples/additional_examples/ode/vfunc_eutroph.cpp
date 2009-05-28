/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     vfunc_eutroph.cpp
 Revision: $Id$
 Contents: example for function module containing the eutroph example
           (based on eutroph.C of version 1.7)
 
   Each << function module >> contains:
          
     (1) const char* const controlFileName 
     (2) int indepDim; 
     (3) int depDim; 
     (4) void initProblemParameters( void )
     (5) void initIndependents( double* indEPS_ )
     (6) void originalVectorFunction( double* indEPS_, double* dEPS_ )
     (7) void tapingVectorFunction( int tag, double* indEPS_, double* dEPS_ )   

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#define _VFUNC_EUTROPH_C_

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <math.h>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "eutrophexam.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;
int depDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */
const double IK    = 0.11 ;
const double FRZ   = 0.3 ;
const double EFFUZ = 0.6;
const double PRITZ = 1.0e-3;
const double RESP  = 5.0e-3;
const double sinK  = 5.0e-3;
const double PRITA = 0.1;
const double RZ    = 1.0e-2;
const double K2    = 4.0e-2;
const double K3    = 5.0e-1;
const double KSP   = 2.0e2;
const double KSF   = 1.0;
const double BETA  = 100.0/1.25;
const double ALPHA = 0.002;
const double TRZ   = 2.0;
const double EPS_P  = 0.4;
const double FI1   = 230.4;
const double FI3   = 282.8;
const double FI4   = 127.5;
const double FI5   = 141.9;
const double p     = 40.0;
const double DEPTH = 45;
const double MORITZ= 0.075;
const double Q     = 0.786E6;
const double VND   = 0.265E9;
/* fix controls */
const double PRFOS = 0.5*p;
const double M     = 0.1;
const double ZMIX  = (45+RZ)/2;
const double QIV   = 0.297E-02/3;


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"EUTROPHEXAM (ADOL-C Example)\n\n");

    /* number of indEPS_ & dEPS_ */
    indepDim = 5;
    depDim   = 5;
}


/****************************************************************************/
/*                                                        INITIALIZE INDEPS_ */
void initIndependents( double* indEPS_ ) {
    indEPS_[0]  = 0.5;
    indEPS_[1]  = 0.0005;
    indEPS_[2]  = 4.0;
    indEPS_[3]  = 0.01; /* originally 0.0 */
    indEPS_[4]  = 0.02; /* originally 0.0 */
}


/****************************************************************************/
/*                                                 ORIGINAL SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                                     The eutroph function */
void eutroph( double* indEPS_, double* dEPS_ ) {
    double T, TEMP, FOTOP, I, PIDI, EPS_, temp, temp2, GROW;
    double V;
    T     = indEPS_[4];
    TEMP  = 9.5+7.9*sin(T+FI1);
    FOTOP = 12.0+4.19*sin(T+280.0);
    I     = 229.0+215.0*sin(T+FI3)+15.3*sin(2.0*T+FI4)+ 21.7*sin(3.0*T+FI5);
    PIDI  = 0.8+.25*cos(T)-.12*cos(2.*T);
    V     = VND;
    if (T < 72)
        I *= 0.603;
    EPS_   = ALPHA * indEPS_[0] + indEPS_[3] + EPS_P;
    temp  = I * exp(-EPS_*ZMIX);
    temp2 = 2*IK*FOTOP;
    GROW  = 1.2*FOTOP/EPS_/ZMIX * (1.333 * atan (I / temp2)
                                   -IK*FOTOP / I * log( 1 + pow((I /temp2 ),2))
                                   -1.333 * atan (temp / temp2)
                                   +IK*FOTOP/temp* log( 1+pow(temp/temp2, 2) ))
            * indEPS_[2] /(KSF+indEPS_[2])
            * 0.366 * pow(K2,0.52) * exp(0.09*TEMP) * pow(indEPS_[0],(1-0.52));
    dEPS_[0] = GROW - RESP * TEMP * indEPS_[0] - FRZ
               * indEPS_[0] * indEPS_[1] - sinK * PIDI * indEPS_[0]
               + (PRITA - indEPS_[0]) * Q/VND;
    dEPS_[1] = FRZ * indEPS_[0] / K2 * indEPS_[1] / 1000
               * EFFUZ*KSP / KSP+indEPS_[0]
               - RZ * indEPS_[1] - MORITZ * indEPS_[1] + (PRITZ - indEPS_[1] ) * Q/V;
    dEPS_[2] = K3 * (-GROW + RESP * TEMP * indEPS_[0] + FRZ * indEPS_[0]
                     * indEPS_[1] * (1 - EFFUZ*KSP /(KSP+indEPS_[0]) ) + RZ * K2 * 1000 *
                     indEPS_[1] + MORITZ * K2 * 1000 * indEPS_[1] )
               + (PRFOS - indEPS_[2])* Q/V;
    dEPS_[3] = (- indEPS_[3] * Q  + BETA * M / TRZ)/VND;
    dEPS_[4] = 1;
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
void originalVectorFunction( double* indEPS_, double* dEPS_ ) {
    eutroph(indEPS_,dEPS_);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                              The active eutroph function */
void activeEutroph( adouble* indEPS_, adouble* dEPS_ ) {
    adouble T, TEMP, FOTOP, I, PIDI, EPS_, temp, temp2, GROW;
    double V;
    T     = indEPS_[4];
    TEMP  = 9.5+7.9*sin(T+FI1);
    FOTOP = 12.0+4.19*sin(T+280.0);
    I     = 229.0+215.0*sin(T+FI3)+15.3*sin(2.0*T+FI4)+ 21.7*sin(3.0*T+FI5);
    PIDI  = 0.8+.25*cos(T)-.12*cos(2.*T);
    V     = VND;
    if (T < 72)
        I *= 0.603;
    EPS_   = ALPHA * indEPS_[0] + indEPS_[3] + EPS_P;
    temp  = I * exp(-EPS_*ZMIX);
    temp2 = 2*IK*FOTOP;
    GROW  = 1.2*FOTOP/EPS_/ZMIX * (1.333 * atan (I / temp2)
                                   -IK*FOTOP / I * log( 1 + pow((I /temp2 ),2))
                                   -1.333 * atan (temp / temp2)
                                   +IK*FOTOP/temp* log( 1+pow(temp/temp2, 2) ))
            * indEPS_[2] /(KSF+indEPS_[2])
            * 0.366 * pow(K2,0.52) * exp(0.09*TEMP) * pow(indEPS_[0],(1-0.52));
    dEPS_[0] = GROW - RESP * TEMP * indEPS_[0] - FRZ
               * indEPS_[0] * indEPS_[1] - sinK * PIDI * indEPS_[0]
               + (PRITA - indEPS_[0]) * Q/VND;
    dEPS_[1] = FRZ * indEPS_[0] / K2 * indEPS_[1] / 1000
               * EFFUZ*KSP / KSP+indEPS_[0]
               - RZ * indEPS_[1] - MORITZ * indEPS_[1] + (PRITZ - indEPS_[1] ) * Q/V;
    dEPS_[2] = K3 * (-GROW + RESP * TEMP * indEPS_[0] + FRZ * indEPS_[0]
                     * indEPS_[1] * (1 - EFFUZ*KSP /(KSP+indEPS_[0]) ) + RZ * K2 * 1000 *
                     indEPS_[1] + MORITZ * K2 * 1000 * indEPS_[1] )
               + (PRFOS - indEPS_[2])* Q/V;
    dEPS_[3] = (- indEPS_[3] * Q  + BETA * M / TRZ)/VND;
    dEPS_[4] = 1;
}


/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
void tapingVectorFunction( int tag, double* indEPS_, double* dEPS_ ) {
    int i;
    trace_on(tag);
    adouble* activeIndEPS_ = new adouble[indepDim];
    adouble* activeDEPS_   = new adouble[depDim];
    adouble* aIP = activeIndEPS_;
    double*  iP  = indEPS_;
    for (i=0; i<indepDim; i++)
        *aIP++ <<= *iP++;
    activeEutroph(activeIndEPS_,activeDEPS_);
    aIP = activeDEPS_;
    iP  = dEPS_;
    for (i=0; i<depDim; i++)
        *aIP++ >>= *iP++;
    trace_off();
}

#undef _VFUNC_EUTROPH_C_





