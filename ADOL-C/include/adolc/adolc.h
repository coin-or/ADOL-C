/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc.h
 Revision: $Id$
 Contents: Provides all C/C++ interfaces of ADOL-C.
           NOTICE: ALL C/C++ headers will be included DEPENDING ON 
           whether the source code is plain C or C/C++ code. 
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
 
----------------------------------------------------------------------------*/

#if !defined(ADOLC_ADOLC_H)
#define ADOLC_ADOLC_H 1

#include <adolc/internal/common.h>

/****************************************************************************/
/*                                                  Now the pure C++ THINGS */
#if defined(__cplusplus)
/*--------------------------------------------------------------------------*/
/* Operator overloading things (active doubles & vectors) */
#  include <adolc/adouble.h>
#  include <adolc/externfcts.h>
#  include <adolc/externfcts2.h>
#  include <adolc/edfclasses.h>
#  include <adolc/checkpointing.h>
#  include <adolc/fixpoint.h>
#endif

/****************************************************************************/
/*                                                     Now the C/C++ THINGS */

/*--------------------------------------------------------------------------*/
/* interfaces to basic forward/reverse routines */
#include <adolc/interfaces.h>

/*--------------------------------------------------------------------------*/
/* interfaces to "Easy To Use" driver routines for ... */
#include <adolc/drivers/drivers.h>    /* optimization & nonlinear equations */
#include <adolc/drivers/taylor.h>     /* higher order tensors & inverse/implicit functions */
#include <adolc/drivers/odedrivers.h> /* ordinary differential equations */
#include <adolc/drivers/psdrivers.h> /* piecewise smooth functions */

/*--------------------------------------------------------------------------*/
/* interfaces to TAPEDOC package */
#include <adolc/tapedoc/tapedoc.h>

/*--------------------------------------------------------------------------*/
/* interfaces to SPARSE package */
#if defined(SPARSE_DRIVERS)
#include <adolc/sparse/sparsedrivers.h>
#include <adolc/sparse/sparse_fo_rev.h>
#endif

/*--------------------------------------------------------------------------*/
/* parameters */
#include <adolc/param.h>

/*--------------------------------------------------------------------------*/
/* tape and value stack utilities */
#include <adolc/taping.h>

/*--------------------------------------------------------------------------*/
/* allocation utilities */
#include <adolc/adalloc.h>

/****************************************************************************/
#endif
