/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     hov_forward_mpi.c
 Revision: $Id$
 Contents: hov_forward_mpi (higher-order-vector parallel forward mode)

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/common.h>
#ifdef HAVE_MPI
#define _MPI_ 1
#define _HOV_ 1
#undef _KEEP_
#include "uni5_for.c"
#undef _HOV_
#undef _MPI_
#endif

