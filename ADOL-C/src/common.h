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

/*--------------------------------------------------------------------------*/
/* standard includes */
#if !defined(__cplusplus)
#   include <stdlib.h>
#   include <stdio.h>
#else
#   include <cstdlib>
#   include <cstdio>
    using namespace std;
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
/* system dependend configuration */
#if defined(ADOLC_INTERNAL)
#   if HAVE_CONFIG_H
#       include <config.h>

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
#           include <malloc.h>
#           define malloc rpl_malloc
#           define calloc rpl_calloc
#       endif /* ADOLC_NO_MALLOC */
#       if defined(ADOLC_NO_REALLOC)
#           include <malloc.h>
#           define realloc rpl_realloc
#       endif /* ADOLC_NO_REALLOC */

#   endif /* HAVE_CONFIG_H */
#endif /* ADOLC_INTERNAL */

/*--------------------------------------------------------------------------*/
/* developer and user parameters */
#include <dvlparms.h>
#include <usrparms.h>

/*--------------------------------------------------------------------------*/
/* windows dll exports/imports */
#if defined(ADOLC_DLL)
#	define ADOLC_DLL_EXPORT __declspec(dllexport)
#else
#	define ADOLC_DLL_EXPORT
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

#define MAXDEC(a,b) if ((a) < (b)) (a) = (b)
#define MINDEC(a,b) if ((a) > (b)) (a) = (b)

#define MAX_ADOLC(a,b) ( (a)<(b)? (b):(a) )
#define MIN_ADOLC(a,b) ( (a)>(b)? (b):(a) )

/*--------------------------------------------------------------------------*/
#endif

