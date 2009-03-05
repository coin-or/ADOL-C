/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fov_offset_forward.c
 Revision: $Id: fov_forward.c 123 2006-07-18 11:35:12Z kowarz $
 Contents: fov_offset_forward (first-order-vector forward mode with
           p-offset in arguments and taylors)

 Copyright (c) Sebastian Schlenkrich
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file. 
 
----------------------------------------------------------------------------*/
#define _FOV_  1
#define _CHUNKED_
#undef _KEEP_
#include <adolc/uni5_for.c>
#undef _CHUNKED_
#undef _FOV_

