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

#include <common.h>

/****************************************************************************/
/*                                                  Now the pure C++ THINGS */
#if defined(__cplusplus)
/*--------------------------------------------------------------------------*/
/* Operator overloading things (active doubles & vectors) */
#  include <adouble.h>
#  include <externfcts.h>
#  include <checkpointing.h>
#  include <fixpoint.h>
#endif

/****************************************************************************/
/*                                                     Now the C/C++ THINGS */

/*--------------------------------------------------------------------------*/
/* interfaces to basic forward/reverse routines */
#include <interfaces.h>

/*--------------------------------------------------------------------------*/
/* interfaces to "Easy To Use" driver routines for ... */
#include <drivers/drivers.h>    /* optimization & nonlinear equations */
#include <drivers/taylor.h>     /* higher order tensors & inverse/implicit functions */
#include <drivers/odedrivers.h> /* ordinary differential equations */

/*--------------------------------------------------------------------------*/
/* interfaces to TAPEDOC package */
#include <tapedoc/tapedoc.h>

/*--------------------------------------------------------------------------*/
/* interfaces to SPARSE package */
#if defined(SPARSE)
#include <sparse/sparsedrivers.h>
#include <sparse/sparse_fo_rev.h>
#endif

/*--------------------------------------------------------------------------*/
/* tape and value stack utilities */
#include <taping.h>

/*--------------------------------------------------------------------------*/
/* allocation utilities */
#include <adalloc.h>

/****************************************************************************/
#endif
