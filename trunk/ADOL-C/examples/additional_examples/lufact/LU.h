/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     LU.h
 Revision: $Id$
 Contents: example for 'active' LU-decomposition and according solver

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
   
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#ifndef _LU_H
#define _LU_H

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>           // use of ALL ADOL-C interfaces


/****************************************************************************/
/* Simple LU-factorization according to Crout's algorithm without pivoting */
void LUfact(int n, adouble **A);


/****************************************************************************/
/* Solution of A*x=b by forward and backward substitution */
void LUsolve(int n, adouble **A, adouble *bx);


/****************************************************************************/
/*                                                              END OF FILE */
#endif
