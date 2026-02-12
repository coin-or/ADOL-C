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

#include <adolc/adolcexport.h>
#include <adolc/internal/common.h>

/* C++ declarations available only when compiling with C++  */
#if defined(__cplusplus)
class ValueTape;
ADOLC_API int lie_scalar(ValueTape &, short, short, double *, short, double *);
ADOLC_API int lie_scalar(ValueTape &, short, short, short, double *, short,
                         double **);
ADOLC_API int lie_gradient(ValueTape &, short, short, double *, short,
                           double **);
ADOLC_API int lie_gradient(ValueTape &, short, short, short, double *, short,
                           double ***);

#endif

/* C-declarations           */
#if defined(__cplusplus)
extern "C" {
#endif

ADOLC_API int lie_scalarc(ValueTape &, ValueTape &, short, double *, short,
                          double *);
ADOLC_API int lie_scalarcv(ValueTape &, ValueTape &, short, short, double *,
                           short, double **);
ADOLC_API int lie_gradientc(ValueTape &, ValueTape &, short, double *, short,
                            double **);
ADOLC_API int lie_gradientcv(ValueTape &, ValueTape &, short, short, double *,
                             short, double ***);
ADOLC_API int lie_covector(ValueTape &, ValueTape &, short, double *, short,
                           double **);
ADOLC_API int lie_bracket(ValueTape &, ValueTape &, short, double *, short,
                          double **);

#if defined(__cplusplus)
}
#endif

#endif
