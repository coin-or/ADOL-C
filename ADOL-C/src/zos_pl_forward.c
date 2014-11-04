/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     zos_an_forward.c
 Revision: $Id$
 Contents: zos_forward (zero-order-scalar abs-normal forward mode)

 Copyright (c) Kshitij Kulshreshtha
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
  
----------------------------------------------------------------------------*/
#define _ZOS_  1
#define _KEEP_ 1
#define _ABS_NORM_ 1
#include <uni5_for.c>
#undef _ABS_NORM_
#undef _KEEP_
#undef _ZOS_

