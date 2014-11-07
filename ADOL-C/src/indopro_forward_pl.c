/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     indopro_forward_pl.c
 Revision: $Id$
 Contents: indopro_forward_pl (index domains abs-normal forward mode)
 
 Copyright (c) Kshitij Kulshrestha
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/
#define _INDO_  1
#define _INDOPRO_ 1
#define _NTIGHT_ 1
#define _ABS_NORM_ 1
#include <uni5_for.c>
#undef _ABS_NORM_
#undef _NTIGHT_
#undef _INDOPRO_
#undef _INDO_
