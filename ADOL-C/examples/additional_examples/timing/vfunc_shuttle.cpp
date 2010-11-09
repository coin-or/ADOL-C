/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     vfunc_shuttle.cpp
 Revision: $Id$
 Contents: Example of function module containing the shuttle example
            (based on shuttlexam.c of version 1.7) 
 
   Each << function module >> contains:
          
     (1) const char* const controlFileName 
     (2) int indepDim; 
     (3) int depDim; 
     (4) void initProblemParameters( void )
     (5) void initIndependents( double* indeps )
     (6) void originalVectorFunction( double* indeps, double* deps )
     (7) void tapingVectorFunction( int tag, double* indeps, double* deps )   
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 
   
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#define _VFUNC_SHUTTLE_C_


/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <cmath>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "shuttlexam.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;
int depDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */
const double Pi = 3.141592654;
const double ae = 20902900.0;
const double mu = 0.14E+17;
const double a  = 40.0;
const double S  = 2690.0;
const double crtd = 180.0/Pi;
const double cl = 0.84-0.48*(38.0-a*crtd)/26.0;
const double C0 =  3.974960446019;
const double C1 = -0.01448947694635;
const double C2 = -0.2156171551995e-4;
const double C3 = -0.1089609507291e-7;
const double V0 =  0.0;
const double ma = 5964.496499824;
const double Om = .72921159e-4;


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"SHUTTLEXAM (ADOL-C Example)\n\n");

    /* number of indeps & deps */
    indepDim = 14;
    depDim   = 7;
}


/****************************************************************************/
/*                                                        INITIALIZE INDEPs */
void initIndependents( double* indeps ) {
    indeps[0]  = 264039.328;   /* H */
    indeps[1]  = 177.718047;   /* x */
    indeps[2]  = 32.0417885;   /* l */
    indeps[3]  = 24317.0798;   /* V */
    indeps[4]  = -0.749986488; /* g */
    indeps[5]  = 62.7883367;   /* A */
    indeps[6]  = 41.100771834; /* b */
    indeps[7]  = -318;         /* Hp */
    indeps[8]  = 0.01;         /* xp */
    indeps[9]  = 0.1;          /* lp */
    indeps[10] = -3.6;         /* Vp */
    indeps[11] = 0.001;        /* gp */
    indeps[12] = 0.1;          /* Ap */
    indeps[13] = 0.06;         /* bp */
}


/****************************************************************************/
/*                                                 ORIGINAL SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                                     The shuttle function */
void shuttle( double* indeps, double* deps ) {
    double r,gr,rho,L,cd,Z;
    double sing,cosg,sinA,cosA,sinl,cosl,tanl;

    r  = indeps[0]+ae;
    gr = mu/(r*r);
    rho= 0.002378*exp(-indeps[0]/23800.0);
    L  = 0.5*rho*cl*S*indeps[3]*indeps[3];
    cd = 0.78-0.58*(38.0-a*crtd)/26.0;
    Z  = .5*rho*cd*S*indeps[3]*indeps[3];
    // evaluate the dynamic equations ...
    sinA = sin(indeps[5]);
    cosA = cos(indeps[5]);
    sing = sin(indeps[4]);
    cosg = cos(indeps[4]);
    sinl = sin(indeps[2]);
    cosl = cos(indeps[2]);
    tanl = sinl/cosl;
    deps[0] = indeps[3]*sing-indeps[7];
    deps[1] = indeps[3]*cosg*sinA/(r*cosl)-indeps[8];
    deps[2] = indeps[3]*cosg*cosA/r-indeps[9];
    deps[3] = -Z/ma-gr*sing-Om*Om*r*cosl
              *(sinl*cosA*cosg-cosl*sing)-indeps[10];
    deps[4] = L*cos(indeps[6])/(ma*indeps[3])
              +cosl/indeps[3]*(indeps[3]*indeps[3]/r-gr)
              +2*Om*cosl*sinA
              +Om*Om*r*cosl/indeps[3]*(sinl*cosA*sing+cosl*cosg)
              -indeps[11];
    deps[5] = L*sin(indeps[6])/(ma*indeps[3]*cosg)+indeps[3]/r*cosg*sinA*tanl
              -2*Om*(cosl*cosA*sing/cosg-sinl)
              +Om*Om*r*cosl*sinl*sinA/(indeps[3]*cosg)-indeps[12];
    deps[6] = Z/ma
              -(C0+(indeps[3]-V0)*(C1+(indeps[3]-V0)*(C2+(indeps[3]-V0)*C3)));
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
void originalVectorFunction( double* indeps, double* deps ) {
    shuttle(indeps,deps);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                              The active shuttle function */
void activeShuttle( adouble* indeps, adouble* deps ) {
    adouble r,gr,rho,L,cd,Z;
    adouble sing,cosg,sinA,cosA,sinl,cosl,tanl;

    r  = indeps[0]+ae;
    gr = mu/(r*r);
    rho= 0.002378*exp(-indeps[0]/23800.0);
    L  = 0.5*rho*cl*S*indeps[3]*indeps[3];
    cd = 0.78-0.58*(38.0-a*crtd)/26.0;
    Z  = .5*rho*cd*S*indeps[3]*indeps[3];
    // evaluate the dynamic equations ...
    sinA = sin(indeps[5]);
    cosA = cos(indeps[5]);
    sing = sin(indeps[4]);
    cosg = cos(indeps[4]);
    sinl = sin(indeps[2]);
    cosl = cos(indeps[2]);
    tanl = sinl/cosl;
    deps[0] = indeps[3]*sing-indeps[7];
    deps[1] = indeps[3]*cosg*sinA/(r*cosl)-indeps[8];
    deps[2] = indeps[3]*cosg*cosA/r-indeps[9];
    deps[3] = -Z/ma-gr*sing-Om*Om*r*cosl
              *(sinl*cosA*cosg-cosl*sing)-indeps[10];
    deps[4] = L*cos(indeps[6])/(ma*indeps[3])
              +cosl/indeps[3]*(indeps[3]*indeps[3]/r-gr)
              +2*Om*cosl*sinA
              +Om*Om*r*cosl/indeps[3]*(sinl*cosA*sing+cosl*cosg)
              -indeps[11];
    deps[5] = L*sin(indeps[6])/(ma*indeps[3]*cosg)+indeps[3]/r*cosg*sinA*tanl
              -2*Om*(cosl*cosA*sing/cosg-sinl)
              +Om*Om*r*cosl*sinl*sinA/(indeps[3]*cosg)-indeps[12];
    deps[6] = Z/ma
              -(C0+(indeps[3]-V0)*(C1+(indeps[3]-V0)*(C2+(indeps[3]-V0)*C3)));
}


/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
void tapingVectorFunction( int tag, double* indeps, double* deps ) {
    int i;
    trace_on(tag);
    adouble* activeIndeps = new adouble[indepDim];
    adouble* activeDeps   = new adouble[depDim];
    adouble* aIP = activeIndeps;
    double*  iP  = indeps;
    for (i=0; i<indepDim; i++)
        *aIP++ <<= *iP++;
    activeShuttle(activeIndeps,activeDeps);
    aIP = activeDeps;
    iP  = deps;
    for (i=0; i<depDim; i++)
        *aIP++ >>= *iP++;
    trace_off();
}

#undef _VFUNC_SHUTTLE_C_





