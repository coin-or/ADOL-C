/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     convolut.h
 Revision: $Id: convolut.h 295 2009-02-25 13:32:25Z awalther $
 Contents: Convolution routines (used by ho_rev.mc)
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_CONVOLUT_H)
#define ADOLC_CONVOLUT_H 1

#include <adolc/common.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                              CONVOLUTION */

/*--------------------------------------------------------------------------*/
/* Evaluates convolution of a and b to c */
void conv( int dim, double *a, revreal *b, double *c );
void conv0( int dim, revreal *a, revreal *b, double *c );

/****************************************************************************/
/*                                                  INCREMENTAL CONVOLUTION */

/*--------------------------------------------------------------------------*/
/* Increments truncated convolution of a and b to c */
void inconv ( int dim, double *a, revreal *b, double* c );

/*--------------------------------------------------------------------------*/
/* Increments truncated convolution of a and b to c and sets a to zero */
void inconv0( int dim, double *a, revreal *b, double* c );
void inconv1( int dim, revreal *a, revreal *b, revreal* c );


/****************************************************************************/
/*                                                  DECREMENTAL CONVOLUTION */

/*--------------------------------------------------------------------------*/
/* Decrements truncated convolution of a and b to c */
void deconv ( int dim, double* a, double *b, double* c );

/*--------------------------------------------------------------------------*/
/* Decrements truncated convolution of a and b to c and sets a to zero */
void deconv0( int dim, double* a, revreal *b, double* c );
void deconv1( int dim, revreal* a, revreal *b, revreal* c );


/****************************************************************************/
/*                                                    OTHER USEFUL ROUTINES */

/*--------------------------------------------------------------------------*/
void divide(int dim, revreal* a, revreal *b, revreal* c);

/*--------------------------------------------------------------------------*/
void recipr(int dim, double  a, revreal *b, revreal* c);


/****************************************************************************/
/*                                                                  ZEROING */

/*--------------------------------------------------------------------------*/
/* Set a to zero */
void zeroset(int dim, double* a);

/*--------------------------------------------------------------------------*/
/* Copies a to tmp and initializes a to zero */
void copyAndZeroset( int dim, double *a, double* tmp);


/****************************************************************************/
END_C_DECLS

#endif
