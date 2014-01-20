/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fos_reverse.c
 Revision: $Id$
 Contents: fos_reverse (first-order-scalar reverse mode)
 
 Copyright (c) Kshitij Kulshreshtha
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#define _FOS_ 1
#define _ABS_NORM_ 1
#include <fo_rev.c>
#undef _ABS_NORM_
#undef _FOS_
