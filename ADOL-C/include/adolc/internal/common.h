/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     common.h
 Revision: $Id$
 Contents: Common (global) ADOL-C header  
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#if !defined(ADOLC_COMMON_H)
#define ADOLC_COMMON_H 1

#include <stdint.h>
/*--------------------------------------------------------------------------*/
/* standard includes */
#if !defined(__cplusplus)
#   include <stdlib.h>
#   include <stdio.h>
#else
#   include <cstdlib>
#   include <cstdio>
#endif

/*--------------------------------------------------------------------------*/
/* type definitions */
    typedef unsigned int uint;

/*--------------------------------------------------------------------------*/
/* OpenMP includes */
#if defined(_OPENMP)
#include <omp.h>
#endif

/*--------------------------------------------------------------------------*/
/* system dependent configuration */
#if defined(ADOLC_INTERNAL)
#   if HAVE_CONFIG_H
#       include "config.h"

/*      malloc/calloc/realloc replacments */
#       undef ADOLC_NO_MALLOC
#       undef ADOLC_NO_REALLOC
#       if !defined(HAVE_MALLOC)
#           define ADOLC_NO_MALLOC 1
#       else
#           if (HAVE_MALLOC == 0)
#               define ADOLC_NO_MALLOC 1
#           endif /* HAVE_MALLOC == 0 */
#       endif /* HAVE_MALLOC */
#       if !defined(HAVE_REALLOC)
#           define ADOLC_NO_REALLOC 1
#       else
#           if (HAVE_REALLOC == 0)
#               define ADOLC_NO_REALLOC 1
#           endif /* HAVE_REALLOC == 0 */
#       endif /* HAVE_REALLOC */

#       if defined(ADOLC_NO_MALLOC)
#           include "rpl_malloc.h"
#           define malloc rpl_malloc
#           define calloc rpl_calloc
#       endif /* ADOLC_NO_MALLOC */
#       if defined(ADOLC_NO_REALLOC)
#           include "rpl_malloc.h"
#           define realloc rpl_realloc
#       endif /* ADOLC_NO_REALLOC */

#       ifndef HAVE_TRUNC
#           define trunc(x) ( (x<0) ? ceil(x) : floor(x) )
#       endif

#   endif /* HAVE_CONFIG_H */
#endif /* ADOLC_INTERNAL */

/*--------------------------------------------------------------------------*/
/* user parameters and settings */
#include <adolc/internal/usrparms.h>
#include <adolc/internal/adolc_settings.h>

/*--------------------------------------------------------------------------*/
/* windows dll exports/imports */
#if defined(ADOLC_DLL)
#   define ADOLC_DLL_EXPORT __declspec(dllexport)
#   define ADOLC_DLL_EXPIMP __declspec(dllexport)
#elif defined(_MSC_VER)
#   define ADOLC_DLL_EXPORT
#   define ADOLC_DLL_EXPIMP __declspec(dllimport)
#else
#   define ADOLC_DLL_EXPORT
#   define ADOLC_DLL_EXPIMP
#endif

/*--------------------------------------------------------------------------*/
/* further helpful macros */
#if defined(__cplusplus)
#  define BEGIN_C_DECLS extern "C" {
#  define END_C_DECLS   }
#else
#  define BEGIN_C_DECLS
#  define END_C_DECLS
#endif

#define MAXDEC(a,b) do { revreal __r = (b); if ( __r > (a) ) (a) = __r; } while (0)
#define MAXDECI(a,b) do { int __r = (b); if ( __r > (a) ) (a) = __r; } while (0)
#define MINDECR(a,b) do { revreal __r = (b); if ( __r < (a) ) (a) = __r; } while (0)
#define MINDEC(a,b) do { int __r = (b); if ( __r < (a) ) (a) = __r; } while (0)

#define MAX_ADOLC(a,b) ( (a)<(b)? (b):(a) )
#define MIN_ADOLC(a,b) ( (a)>(b)? (b):(a) )

/*--------------------------------------------------------------------------*/
#endif

