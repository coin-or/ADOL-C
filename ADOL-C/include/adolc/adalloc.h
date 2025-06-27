/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adalloc.h
 Revision: $Id$
 Contents: Allocation of arrays of doubles in several dimensions

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#if !defined(ADOLC_ADALLOC_H)
#define ADOLC_ADALLOC_H 1

#include <adolc/adolcexport.h>
#include <adolc/internal/common.h>

/****************************************************************************/
/*                                                         Now the C THINGS */
BEGIN_C_DECLS

/*--------------------------------------------------------------------------*/
/*                                              MEMORY MANAGEMENT UTILITIES */
ADOLC_API char *populate_dpp(double ***const pointer, char *const memory, int n,
                             int m);
ADOLC_API char *populate_dppp(double ****const pointer, char *const memory,
                              int n, int m, int p);
ADOLC_API char *populate_dppp_nodata(double ****const pointer,
                                     char *const memory, int n, int m);
ADOLC_API double *myalloc1(size_t);
ADOLC_API double **myalloc2(size_t, size_t);
ADOLC_API double ***myalloc3(size_t, size_t, size_t);

ADOLC_API void myfree1(double *);
ADOLC_API void myfree2(double **);
ADOLC_API void myfree3(double ***);

/*--------------------------------------------------------------------------*/
/*                                          SPECIAL IDENTITY REPRESENTATION */
ADOLC_API double **myallocI2(int);
ADOLC_API void myfreeI2(int, double **);

ADOLC_API unsigned int *myalloc1_uint(int);

ADOLC_API size_t *myalloc1_ulong(int);
ADOLC_API size_t **myalloc2_ulong(int, int);

/****************************************************************************/
/*                              INTEGER VARIANT FOR BIT PATTERN PROPAGATION */

ADOLC_API void myfree1_uint(unsigned int *);

ADOLC_API void myfree1_ulong(size_t *);
ADOLC_API void myfree2_ulong(size_t **);

END_C_DECLS

/****************************************************************************/
/*                                                       Now the C++ THINGS */
#if defined(__cplusplus)

/*--------------------------------------------------------------------------*/
/*                                              MEMORY MANAGEMENT UTILITIES */
ADOLC_API inline double *myalloc(int n) { return myalloc1(n); }
ADOLC_API inline double **myalloc(int m, int n) { return myalloc2(m, n); }
ADOLC_API inline double ***myalloc(int m, int n, int p) {
  return myalloc3(m, n, p);
}

ADOLC_API inline void myfree(double *A) { myfree1(A); }
ADOLC_API inline void myfree(double **A) { myfree2(A); }
ADOLC_API inline void myfree(double ***A) { myfree3(A); }

#endif

/****************************************************************************/
#endif
