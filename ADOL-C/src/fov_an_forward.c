/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fov_an_forward.c
 Revision: $Id$
 Contents: fov_an_forward (first-order-vector abs-normal forward mode)
 
 Copyright (c) Kshitij Kulshrestha
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/
#define _FOV_  1
#undef _KEEP_
#define _ABS_NORM_ 1
#include <uni5_for.c>
#undef _ABS_NORM_
#undef _FOV_

