/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     malloc.c
 Revision: $Id$
 Contents: malloc replacements for not gnu compatible malloc system functions
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#include "rpl_malloc.h"

#undef ADOLC_NO_MALLOC
#undef ADOLC_NO_REALLOC
#if defined(ADOLC_INTERNAL)
#   if !defined(HAVE_MALLOC)
#       define ADOLC_NO_MALLOC 1
#   else
#       if (HAVE_MALLOC == 0)
#           define ADOLC_NO_MALLOC 1
#       endif /* HAVE_MALLOC == 0 */
#   endif /* HAVE_MALLOC */
#   if !defined(HAVE_REALLOC)
#       define ADOLC_NO_REALLOC 1
#   else
#       if (HAVE_REALLOC == 0)
#           define ADOLC_NO_REALLOC 1
#       endif /* HAVE_REALLOC == 0 */
#   endif /* HAVE_REALLOC */
#endif /* ADOLC_INTERNAL */

#if defined(ADOLC_NO_MALLOC)
#   undef malloc
#   undef calloc
    extern void *malloc();
    extern void *calloc();

    /** Allocate an n-byte block from the heap! n>=1
     * If native malloc(0) returns an invalid pointer use the
     * replacement-function instead.
     */
    void *rpl_malloc(size_t n) {
        if (n == 0) n = 1;
        return malloc(n);
    }

    void *rpl_calloc(size_t n, size_t size) {
        if (n == 0) n = 1;
        if (size == 0) size = 1;
        return calloc(n, size);
    }

#endif /* ADOLC_NO_MALLOC */

#if defined(ADOLC_NO_REALLOC)
#   undef realloc
    extern void *realloc();

    void *rpl_realloc(void *ptr, size_t size) {
        if (size == 0) size = 1;
        if (ptr == NULL) ptr = rpl_malloc(1);
        return realloc(ptr, size);
    }
#endif /* ADOLC_NO_REALLOC */
