/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sfunc_michalewitz.cpp
 Revision: $Id$
 Contents: function module containing Michalewitz' function

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
#define _SFUNC_MICHALEWITZ_C_


/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <cmath>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "michalewitzexam.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */
#define Pi 3.141592654
const double M  = 10.0;


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"MICHALEWITZ' FUNCTION (ADOL-C Example)\n\n");
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
    for (i=0; i<indepDim; i++)
        indeps[i] = Pi*(i+1.0)/(2.0+i);
}


/****************************************************************************/
/*                                                 ORIGINAL SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                                    Michalewitz' function */
double micha( int dim, double* indeps ) {
    int i;
    double u = 0;
    for (i=0; i<dim; i++)
        u += sin(indeps[i]) * pow(sin((i+1)*indeps[i]*indeps[i]/Pi),2.0*M);
    return -u;
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
double originalScalarFunction( double* indeps ) {
    return micha(indepDim, indeps);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                             active Michalewitz' function */
adouble activeMicha( int dim, adouble* indeps ) {
    int i;
    adouble u = 0;
    for (i=0; i<dim; i++)
        u += sin(indeps[i]) * pow(sin((i+1)*indeps[i]*indeps[i]/Pi),2.0*M);
    return -u;
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
    adouble ares = activeMicha(indepDim, activeIndeps);
    double res = 0;
    ares >>= res;
    trace_off();
    return res;
}

#undef _SFUNC_MICHALEWITZ_C_





