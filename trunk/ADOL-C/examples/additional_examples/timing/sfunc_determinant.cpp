/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sfunc_determinant.cpp
 Revision: $Id$
 Contents: function module containing the determinant example
 
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
#define _SFUNC_DETERMINANT_C_

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "detexam.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */
int matrixDim;
int mRec;


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"COMPUTATION OF DETERMINANTS Type 2 (ADOL-C Example)\n\n");
    if (indepDim > 0)
        matrixDim = indepDim;
    else {
        fprintf(stdout,"    order of matrix = ? ");
        fscanf(stdin,"%d",&matrixDim);
        fprintf(stdout,"\n");
    }
    indepDim = matrixDim * matrixDim;
}


/****************************************************************************/
/*                                                        INITIALIZE INDEPs */
void initIndependents( double* indeps ) {
    int i, j;
    double* iP = indeps;
    mRec = 1;
    for (i=0; i<matrixDim; i++) {
        mRec *= 2;
        for (j=0; j<matrixDim; j++)
            *iP++ = j/(1.0+i);
        indeps[i*matrixDim+i] += 1.0;
    }
}


/****************************************************************************/
/*                                                 ORIGINAL SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                       The recursive determinant function */
double det( int k, int m, double* indeps ) {
    int i;
    if (m == 0)
        return 1.0;
    else {
        double* pt = indeps+((k-1)*matrixDim);
        double  t  = 0;
        int     p  = 1;
        int     p1, s;
        if (k%2)
            s = 1;
        else
            s = -1;
        for (i=0; i<matrixDim; i++) {
            p1 = 2*p;
            if (m%p1 >= p) {
                if (m == p) {
                    if (s > 0)
                        t += *pt;
                    else
                        t -= *pt;
                } else {
                    if (s > 0)
                        t += (*pt)*det(k-1, m-p, indeps);
                    else
                        t -= (*pt)*det(k-1, m-p, indeps);
                }
                s = -s;
            }
            ++pt;
            p = p1;
        }
        return t;
    }
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
double originalScalarFunction( double* indeps ) {
    return det(matrixDim, mRec-1, indeps);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                       The recursive determinant function */
adouble activeDet( int k, int m, adouble* indeps ) {
    int i;
    if (m == 0)
        return 1.0;
    else {
        adouble* pt = indeps + ((k-1)*matrixDim);
        adouble  t  = 0;
        int      p  = 1;
        int      p1, s;
        if (k%2)
            s = 1;
        else
            s = -1;
        for (i=0; i<matrixDim; i++) {
            p1 = 2*p;
            if (m%p1 >= p) {
                if (m == p) {
                    if (s > 0)
                        t += *pt;
                    else
                        t -= *pt;
                } else {
                    if (s > 0)
                        t += (*pt)*activeDet(k-1, m-p, indeps);
                    else
                        t -= (*pt)*activeDet(k-1, m-p, indeps);
                }
                s = -s;
            }
            ++pt;
            p = p1;
        }
        return t;
    }
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
    adouble ares = activeDet(matrixDim, mRec-1, activeIndeps);
    double res = 0;
    ares >>= res;
    trace_off();
    return res;
}

#undef _SFUNC_DETERMINANT_C_





