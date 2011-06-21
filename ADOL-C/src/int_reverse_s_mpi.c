/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     int_reverse_s_mpi.c
 Revision: $Id$
 Contents: int_reverse_mpi (integer parallel reverse mode safe for bit pattern propagation)

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/common.h>
#ifdef HAVE_MPI
#define _MPI_ 1
#define _INT_REV_ 1
#define _NTIGHT_ 1
#include  "fo_rev.c"
#undef _INT_REV_
#undef _NTIGHT_
#undef _MPI_
#endif

