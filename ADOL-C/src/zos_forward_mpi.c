/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     zos_forward_mpi.c
 Revision: $Id$
 Contents: zos_forward_mpi (zero-order-scalar parallel forward mode)

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/common.h>
#ifdef HAVE_MPI
#define _MPI_  1
#define _ZOS_  1
#define _KEEP_ 1
#include "uni5_for.c"
#undef _KEEP_
#undef _ZOS_
#undef _MPI_
#endif

