/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     createTape.cpp
 Revision: $Id$
 Contents:

   How to create a tape with ADOL-C
   ================================
   
   Application of ADOL-C to the example function of Tony Wong
   to obtain the tape, which will be saved on HD for future use
 
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

/****************************************************************************/
/*                                                     ACTIVATING FUNCTIONS */


/* Example of a C function */
double myf ( double x, double y, double z) {
    return x*x + y*y*y + z*z*z*z + 2.0*x*x*x*x*y*y*y
           + z*z*z*z*x*x*x*x*x + 3.0*z*z*z*y;
}


/* Example of the corresponding 'active' C function */
adouble myf ( adouble x, adouble y, adouble z) {
    return x*x + y*y*y + z*z*z*z + 2.0*x*x*x*x*y*y*y
           + z*z*z*z*x*x*x*x*x + 3.0*z*z*z*y;

}


/****************************************************************************/
/*                                           GENERATING THE EVALUATION TAPE */


int main() {
    double  xyz[3],  f;              /* variables */
    adouble ax, ay, az, af;          /* active varaibles */

    xyz[0] = 1.2;
    xyz[1] = 2.6;      /* initialize any values */
    xyz[2] = 0.03;


    /* TRACING THE EVALUATION TAPE */
    trace_on(1);                     /* start tracing of an evaluation
                                                  tape with the identifier 1 */
    ax <<= xyz[0];                   /* marking independent variables */
    ay <<= xyz[1];
    az <<= xyz[2];
    af = myf(ax,ay,az);              /* calling the 'active' version of
                                                  the function to be differentiated
                                                  to generate a tape of the evaluation
                                                  process;
                                                  NOTE: Instead of calling a
                                                  C function the whole evaluation code
                                                  can be placed here (see example file
                                                  DEX/powerexam.C) */
    af >>= f;                        /* marking the only one dependent
                                                  variable */
    trace_off(1);                    /* stop tracing */

    /* NOTE: trace_off(..) is called with the value 1 (for the optional
       !     argument). This forces ADOL-C to save the generated tapes
       !     on harddisc. In particular these are the files
       !
       !        _adol-op_tape.1      (operations  = opcodes)
       !        _adol-in_tape.1      (integers    = locations)
       !        _adol-rl_tape.1      (real values = doubles)
       !
       !     The appendix '1' is determined by the used tape
       !     identifier, which was passed to trace_on(..).
    */
}

















