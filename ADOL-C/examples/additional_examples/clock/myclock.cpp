/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     myclock.cpp
 Revision: $Id$
 Contents: timing utilities

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
 
/****************************************************************************/
/*                                                                 INCLUDES */
#include <sys/timeb.h>
#include <time.h>
#include "myclock.h"



/****************************************************************************/
/*                                                          CLOCK UTILITIES */

double myclock( int normalize ) {
    struct timeb tb;

    ftime(&tb);
    return ((double)tb.time+(double)tb.millitm/1000.);
}

void normalize() {}

