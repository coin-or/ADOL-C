/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     hos_forward_mpi.c
 Revision: $Id$
 Contents: hos_forward_mpi (higher-order-scalar parallel forward mode)

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/common.h>
#ifdef HAVE_MPI
#define _MPI_  1
#define _HOS_  1
#define _KEEP_ 1
#include "uni5_for.c"
#undef _KEEP_
#undef _HOS_
#undef _MPI_
#endif

