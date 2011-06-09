/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/drivers.h
 Revision: $Id$
 Contents: hos_ov_reverse (higher-order-scalar reverse mode on vectors)
 
 Copyright (c) Andrea Walther
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#define _HOS_OV_ 1
#include <ho_rev.c>
#undef _HOS_OV_
