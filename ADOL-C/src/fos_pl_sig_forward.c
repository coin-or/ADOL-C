/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fos_an_forward.c
 Revision: $Id$
 Contents: fos_an_forward (first-order-scalar abs-normal forward mode)

 Copyright (c) Kshitij Kulshreshtha
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
  
----------------------------------------------------------------------------*/
#define _FOS_  1
#undef _KEEP_
#define _ABS_NORM_SIG_ 1
#include <uni5_for.c>
#undef _ABS_NORM_SIG_
#undef _FOS_
