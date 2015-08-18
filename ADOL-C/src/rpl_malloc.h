/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     malloc.h
 Revision: $Id$
 Contents: malloc replacements for not gnu compatible malloc system functions

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
  
----------------------------------------------------------------------------*/

#if !defined(ADOLC_MALLOC_H)
#   define ADOLC_MALLOC_H 1

#   if defined(ADOLC_INTERNAL)
#       if defined(HAVE_CONFIG_H)
#           include "config.h"

#           undef ADOLC_NO_MALLOC
#           undef ADOLC_NO_REALLOC
#           if !defined(HAVE_MALLOC)
#               define ADOLC_NO_MALLOC 1
#           else
#               if (HAVE_MALLOC == 0)
#                   define ADOLC_NO_MALLOC 1
#               endif /* HAVE_MALLOC == 0 */
#           endif /* HAVE_MALLOC */
#           if !defined(HAVE_REALLOC)
#               define ADOLC_NO_REALLOC 1
#           else
#               if (HAVE_REALLOC == 0)
#                   define ADOLC_NO_REALLOC 1
#               endif /* HAVE_REALLOC == 0 */
#           endif /* HAVE_REALLOC */

#           if defined(ADOLC_NO_MALLOC)
#               include <stddef.h>
#               if defined(__cplusplus)
                    extern "C" {
#               endif /* __cplusplus */
#               undef rpl_malloc
#               undef rpl_calloc
                extern void *rpl_malloc(size_t);
                extern void *rpl_calloc(size_t, size_t);
#               if defined(__cplusplus)
                    }
#               endif /* __cplusplus */
#           endif /* ADOLC_NO_MALLOC */

#           if defined(ADOLC_NO_REALLOC)
#               include <stddef.h>
#               if defined(__cplusplus)
                    extern "C" {
#               endif /* __cplusplus */
#               undef rpl_realloc
                extern void *rpl_realloc(void *, size_t);
#               if defined(__cplusplus)
                    }
#               endif /* __cplusplus */
#           endif /* ADOLC_NO_REALLOC */

#       endif /* HAVE_CONFIG_H */
#   endif /* ADOLC_INTERNAL */
#endif /* ADOLC_MALLOC_H */
