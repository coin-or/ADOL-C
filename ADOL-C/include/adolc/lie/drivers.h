/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     lie/drivers.h
 Revision: $Id$
 Contents: functions for computation of Lie derivatives
 

 Copyright (c) Siquian Wang, Klaus Röbenack, Jan Winkler, Mirko Franke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
  
----------------------------------------------------------------------------*/
#if !defined(ADOLC_LIE_DRIVER_H)
#define ADOLC_LIE_DRIVER_H

#include "adolc/internal/common.h"


// C++ declarations available only when compiling with C++
#if defined(__cplusplus)

ADOLC_DLL_EXPORT int lie_scalar(short, short, short, double*, short, double*);
ADOLC_DLL_EXPORT int lie_scalar(short, short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_gradient(short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_gradient(short, short, short, short, double*, short, double***);

#endif



// C-declarations
#if defined (__cplusplus)
extern "C" {
#endif
 
ADOLC_DLL_EXPORT int lie_scalarc(short, short, short, double*, short, double*);
ADOLC_DLL_EXPORT int lie_scalarcv(short, short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_gradientc(short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_gradientcv(short, short, short, short, double*, short, double***);
ADOLC_DLL_EXPORT int lie_covector(short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_bracket(short, short, short, double*, short, double**);

#if defined (__cplusplus)
}
#endif



#endif

