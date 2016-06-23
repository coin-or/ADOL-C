/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     tapedoc/ascii2tape.h
 Revision: $Id$
 Contents: Routine ascii2trace(..) converts an ascii description of the trace
           to a real trace in ADOL-C core or disk

 Copyright (c) Kshitij Kulshreshtha


 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_TAPEDOC_ASCIITAPES_H)
#define ADOLC_TAPEDOC_ASCIITAPES_H 1

#include <adolc/internal/common.h>

BEGIN_C_DECLS


/****************************************************************************/
/*                                                         read_ascii_trace */

ADOLC_DLL_EXPORT void read_ascii_trace(const char *const fname, short tag);

/****************************************************************************/
/*                                                        write_ascii_trace */

ADOLC_DLL_EXPORT void write_ascii_trace(const char *const fname, short tag);

/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS

#endif
