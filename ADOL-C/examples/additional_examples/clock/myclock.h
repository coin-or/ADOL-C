/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     myclock.h
 Revision: $Id$
 Contents: timing utilities

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#ifndef _MYCLOCK_H_
#define _MYCLOCK_H_

/****************************************************************************/
/*                                                        CLOCKS PER SECOND */
extern double clocksPerSecond;


/****************************************************************************/
/*                                                                    CLOCK */
double myclock(int normalize = 0);


/****************************************************************************/
/*                                                          NORMALIZE CLOCK */
void normalizeMyclock( void );

#endif








