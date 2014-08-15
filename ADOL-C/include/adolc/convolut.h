/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     convolut.h
 Revision: $Id$
 Contents: Convolution routines (used by ho_rev.mc)
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_CONVOLUT_H)
#define ADOLC_CONVOLUT_H 1

#include <adolc/internal/common.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                              CONVOLUTION */

/*--------------------------------------------------------------------------*/
/* Evaluates convolution of a and b to c */
void conv( int dim, revreal *a, revreal *b, revreal *c );
void conv0( int dim, revreal *a, revreal *b, revreal *c );

/****************************************************************************/
/*                                                  INCREMENTAL CONVOLUTION */

/*--------------------------------------------------------------------------*/
/* Increments truncated convolution of a and b to c */
void inconv ( int dim, revreal *a, revreal *b, revreal* c );

/*--------------------------------------------------------------------------*/
/* Increments truncated convolution of a and b to c and sets a to zero */
void inconv0( int dim, revreal *a, revreal *b, revreal* c );
void inconv1( int dim, revreal *a, revreal *b, revreal* c );


/****************************************************************************/
/*                                                  DECREMENTAL CONVOLUTION */

/*--------------------------------------------------------------------------*/
/* Decrements truncated convolution of a and b to c */
void deconv ( int dim, revreal* a, revreal *b, revreal* c );

/*--------------------------------------------------------------------------*/
/* Decrements truncated convolution of a and b to c and sets a to zero */
void deconv0( int dim, revreal* a, revreal *b, revreal* c );
void deconv1( int dim, revreal* a, revreal *b, revreal* c );
void deconvZeroR( int dim, revreal *a, revreal *b, revreal *c );


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
void copyAndZeroset( int dim, revreal *a, revreal* tmp);


/****************************************************************************/
END_C_DECLS

#endif
