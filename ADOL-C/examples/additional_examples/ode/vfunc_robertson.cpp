/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     vfunc_robertson.cpp
 Revision: $Id$
 Contents: example for function module containing the Robertson test problem
            (based on odexam.C of version 1.7)
 
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
#define _VFUNC_ROBERTSON_C_

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <math.h>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "robertsonexam.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;
int depDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"ROBERTSONEXAM (ADOL-C Example)\n\n");

    /* number of indeps & deps */
    indepDim = 3;
    depDim   = 3;
}


/****************************************************************************/
/*                                                        INITIALIZE INDEPs */
void initIndependents( double* indeps ) {
    indeps[0]  = 1.0;
    indeps[1]  = 0.01; /* originally 0.0 */
    indeps[2]  = 0.02; /* originally 0.0 */
}


/****************************************************************************/
/*                                                 ORIGINAL SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                               The Robertson test problem */
void robertson( double* indeps, double* deps ) {
    deps[0] = -sin(indeps[2]) + 1.0e8*indeps[2]*(1.0-1.0/indeps[0]);
    deps[1] = -10.0*indeps[0]
              + 3.0e7*indeps[2]*(1.0-indeps[1]);
    deps[2] = -deps[0] - deps[1];
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
void originalVectorFunction( double* indeps, double* deps ) {
    robertson(indeps,deps);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                        The active Robertson test problem */
void activeRobertson( adouble* indeps, adouble* deps ) {
    deps[0] = -sin(indeps[2]) + 1.0e8*indeps[2]*(1.0-1.0/indeps[0]);
    deps[1] = -10.0*indeps[0]
              + 3.0e7*indeps[2]*(1.0-indeps[1]);
    deps[2] = -deps[0] - deps[1];
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
    activeRobertson(activeIndeps,activeDeps);
    aIP = activeDeps;
    iP  = deps;
    for (i=0; i<depDim; i++)
        *aIP++ >>= *iP++;
    trace_off();
}

#undef _VFUNC_ROBERTSON_C_





