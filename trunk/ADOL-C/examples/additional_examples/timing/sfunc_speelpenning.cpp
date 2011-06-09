/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sfunc_speelpenning.cpp
 Revision: $Id$
 Contents: function module containing Speepennings product

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
#define _SFUNC_SPEELPENNING_C_


/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <time.h>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "speelpenning.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"SPEELPENNINGS PRODUCT Type 2 (ADOL-C Example)\n\n");
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
/*                                                    Speelpennings product */
double speelpenning(int dim, double* indeps ) {
    int i;
    double y = 1.0;
    for (i=0; i<dim; i++)
        y *= indeps[i];
    return y;
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
double originalScalarFunction( double* indeps ) {
    return speelpenning(indepDim, indeps);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                             active Speelpennings product */
adouble activeSpeelpenning(int dim, adouble* indeps ) {
    int i;
    adouble y = 1.0;
    for (i=0; i<dim; i++)
        y *= indeps[i];
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
    adouble ares = activeSpeelpenning(indepDim, activeIndeps);
    double res = 0;
    ares >>= res;
    trace_off();
    return res;
}

#undef _SFUNC_SPEELPENNING_C_





