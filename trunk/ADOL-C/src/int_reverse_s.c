/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     int_reverse_s.c
 Revision: $Id$
 Contents: int_reverse (integer reverse mode safe for bit pattern propagation)
 
 Copyright (c) Andrea Walther, Christo Mitev
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/
#define _INT_REV_ 1
#define _NTIGHT_ 1
#include  <fo_rev.c>
#undef _INT_REV_
#undef _NTIGHT_

