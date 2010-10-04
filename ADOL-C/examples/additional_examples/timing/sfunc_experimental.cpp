/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sfunc_experimental.cpp
 Revision: $Id$
 Contents: function module containing an experimental function
 
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
#define _SFUNC_EXPERIMENTAL_C_


/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <time.h>
#include <cmath>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "experimental.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */
//#define CODE sqrt(indeps[i])
#define CODE sin(indeps[i])
//#define CODE indeps[i]*indeps[i]


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"EXPERIMENTAL EXAMPLE (ADOL-C Example)\n\n");
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
        indeps[i] = (i+1.0)/(2.0+i);
}


/****************************************************************************/
/*                                                 ORIGINAL SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                                    experimental function */
double experimental( int dim, double* indeps ) {
    int i;
    double y = 1.0;
    for (i=0; i<dim; i++)
        y += CODE;
    return y;
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
double originalScalarFunction( double* indeps ) {
    return experimental(indepDim, indeps);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                             active experimental function */
adouble activeExperimental( int dim, adouble* indeps ) {
    int i;
    adouble y = 1.0;
    for (i=0; i<dim; i++)
        y += CODE;
    return y;
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
    adouble ares = activeExperimental(indepDim, activeIndeps);
    double res = 0;
    ares >>= res;
    trace_off();
    return res;
}

#undef _SFUNC_EXPERIMENTAL_C_







