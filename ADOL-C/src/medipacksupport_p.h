/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     ampisupportAdolc.cpp
 Revision: $Id$

 Copyright (c) Jean Utke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_MEDISUPPORTADOLCP_H)
#define ADOLC_MEDISUPPORTADOLCP_H 1

#ifdef ADOLC_MEDIPACK_SUPPORT

#if defined(__cplusplus)
#include <adolc/medipacksupport.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif
  void mediCallHandleReverse(short tapeId, locint index, double* primalVec, double** adjointVec, int vecSize);
  void mediCallHandleForward(short tapeId, locint index, double* primalVec, double** adjointVec, int vecSize);
  void mediCallHandlePrimal(short tapeId, locint index, double* primalVec);
#if defined(__cplusplus)
}
#endif
void mediInitTape(short tapeId);

#endif

#endif
