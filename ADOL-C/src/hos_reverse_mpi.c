/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     hos_reverse_mpi.c
 Revision: $Id$
 Contents: hos_reverse_mpi (higher-order-scalar parallel reverse mode)

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/common.h>
#ifdef HAVE_MPI
#define _MPI_ 1
#define _HOS_ 1
#include "ho_rev.c"
#undef _HOS_
#undef _MPI_
#endif

