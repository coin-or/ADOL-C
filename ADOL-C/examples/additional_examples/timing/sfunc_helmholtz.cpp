/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sfunc_helmholtz.cpp
 Revision: $Id$
 Contents: function module containing Helmholtz energy function

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
#define _SFUNC_HELMHOLTZ_C_


/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>
#include <cmath>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "helmholtzexam.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */
double *bv = NULL;
const double R = sqrt(2.0);
const double TE= 0.01; /* originally 0.0 */


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"HELMHOLTZ ENERGY (ADOL-C Example)\n\n");
    if (indepDim <= 0) {
        fprintf(stdout,"    number of independent variables = ? ");
        fscanf(stdin,"%d",&indepDim);
        fprintf(stdout,"\n");
    }
}


/****************************************************************************/
/*                                                        INITIALIZE INDEPs */
void initIndependents( double* indeps ) {
    int i;
    double r = 1.0/indepDim;
    if (bv)
        delete[] bv;
    bv = new double[indepDim];
    for (i=0; i<indepDim; i++) {
        indeps[i] = r*sqrt(1.0+i);
        bv[i]     = 0.02*(1.0+fabs(sin(double(i))));
    }
}


/****************************************************************************/
/*                                                 ORIGINAL SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                                         Helmholtz energy */
double helmholtz( int dim, double* indeps, double* bv ) {
    int i,j;
    double he;
    double xax, bx, tem;
    xax = 0;
    bx  = 0;
    he  = 0;
    for (i=0; i<dim; i++) {
        he += indeps[i]*log(indeps[i]);
        bx += bv[i]*indeps[i];
        tem = (2.0/(1.0+i+i))*indeps[i];
        for (j=0; j<i; j++)
            tem += (1.0/(1.0+i+j))*indeps[j];
        xax += indeps[i]*tem;
    }
    xax *= 0.5;
    he   = 1.3625E-3*(he-TE*log(1.0-bx));
    he   = he - log((1+bx*(1+R))/(1+bx*(1-R)))*xax/bx;
    return he;
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
double originalScalarFunction( double* indeps ) {
    return helmholtz(indepDim, indeps, bv);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                                  active Helmholtz energy */
adouble activeHelmholtz( int dim, adouble* indeps, double* bv ) {
    int i,j;
    adouble he;
    adouble xax, bx, tem;
    xax = 0;
    bx  = 0;
    he  = 0;
    for (i=0; i<dim; i++) {
        he += indeps[i]*log(indeps[i]);
        bx += bv[i]*indeps[i];
        tem = (2.0/(1.0+i+i))*indeps[i];
        for (j=0; j<i; j++)
            tem += (1.0/(1.0+i+j))*indeps[j];
        xax += indeps[i]*tem;
    }
    xax *= 0.5;
    he   = 1.3625E-3*(he-TE*log(1.0-bx));
    he   = he - log((1+bx*(1+R))/(1+bx*(1-R)))*xax/bx;
    return he;
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
double tapingScalarFunction( int tag, double* indeps ) {
    int i;
    trace_on(tag);
    adouble* activeIndeps = new adouble[indepDim];
    adouble* aIP = activeIndeps;
    double*  iP  = indeps;
    for (i=0; i<indepDim; i++)
        *aIP++ <<= *iP++;
    adouble ares = activeHelmholtz(indepDim, activeIndeps, bv);
    double res = 0;
    ares >>= res;
    trace_off();
    return res;
}

#undef _SFUNC_HELMHOLTZ_C_





