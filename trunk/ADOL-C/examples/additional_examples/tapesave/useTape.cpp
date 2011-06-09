/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     createTape.cpp
 Revision: $Id$
 Contents:

   How to use a tape with ADOL-C
   =============================
   
   Application of ADOL-C to the example function of Tony Wong
   to use a tape, which was previously saved on HD
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
/* use of ALL ADOL-C interfaces */
#include <adolc/adolc.h>

#include <cstdio>


/****************************************************************************/
/*                                                USING THE EVALUATION TAPE */


int main() {
    double  xyz[3],  f, gradf[3];    /* variables */

    xyz[0] = 1.0;
    xyz[1] = 1.0;      /* initialize any values */
    xyz[2] = 1.0;

    /* USING THE TAPE FOR DERIVATIVE COMPUTATION */

    function(1,                      /* the tape identifier 1 (tag) */
             1,                      /* number of dependent variables
                                                                             = dimension of f */
             3,                      /* number of independent variables
                                                                             = dimension of xyz */
             xyz,                    /* the point where the function has
                                                                             to be differentiated */
             &f);                    /* contains the function value (after
                                                  calling 'function(..)') */

    gradient(1,                      /* the tape identifier 1 (tag) */
             3,                      /* number of independent variables
                                                                             = dimension of xyz */
             xyz,                    /* the point where the function has
                                                                             to be differentiated */
             gradf);                 /* contains the gradient (after
                                                  calling 'gradient(..)') */

    /* print the results */
    fprintf(stdout,"f  = %E\n",f);
    fprintf(stdout,"df = (%E, %E, %E)\n",gradf[0],gradf[1],gradf[2]);

    return 0;
}

















