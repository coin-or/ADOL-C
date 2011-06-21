/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     nonl_ind_forward_s_mpi.c
 Revision: $Id$
 Contents: nonl_ind_forward_s_mpi (parallel propagation of nonlinear index domains)

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/common.h>
#ifdef HAVE_MPI
#define _MPI_ 1
#define _INDO_ 1
#define _NONLIND_ 1
#define _NTIGHT_ 1
#include  "uni5_for.c"
#undef _INDO_
#undef _NONLIND_
#undef _NTIGHT_
#undef _MPI_
#endif

