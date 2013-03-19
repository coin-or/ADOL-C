/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fov_reverse_mpi.c
 Revision: $Id$
 Contents: fov_reverse_mpi (first-order-vector parallel reverse mode)

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/common.h>
#ifdef HAVE_MPI
#define _MPI_ 1
#define _FOV_ 1
#include "fo_rev.c"
#undef _FOV_
#undef _MPI_
#endif

