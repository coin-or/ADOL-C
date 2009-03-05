/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     int_forward_s.c
 Revision: $Id: int_forward_s.c 149 2006-11-08 13:19:25Z kowarz $
 Contents: int_forward (integer forward mode safe for bit pattern propagation)
 
 Copyright (c) Andrea Walther, Christo Mitev
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/
#define _INT_FOR_ 1
#define _NTIGHT_ 1
#include  <adolc/uni5_for.c>
#undef _INT_FOR_
#undef _NTIGHT_

