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
#if !defined (ADOLC_ADALLOC_H)
#define ADOLC_ADALLOC_H 1

#include <adolc/internal/common.h>

/****************************************************************************/
/*                                                         Now the C THINGS */
BEGIN_C_DECLS

/*--------------------------------------------------------------------------*/
/*                                              MEMORY MANAGEMENT UTILITIES */
ADOLC_DLL_EXPORT char* populate_dpp(double ***const pointer, char *const memory,
                                    int n, int m);
ADOLC_DLL_EXPORT char* populate_dppp(double ****const pointer, char *const memory,
                                     int n, int m, int p);
ADOLC_DLL_EXPORT char* populate_dppp_nodata(double ****const pointer, char *const memory,
                                            int n, int m);
ADOLC_DLL_EXPORT double    *myalloc1(size_t);
ADOLC_DLL_EXPORT double   **myalloc2(size_t, size_t);
ADOLC_DLL_EXPORT double  ***myalloc3(size_t, size_t, size_t);

ADOLC_DLL_EXPORT void myfree1(double   *);
ADOLC_DLL_EXPORT void myfree2(double  **);
ADOLC_DLL_EXPORT void myfree3(double ***);

/*--------------------------------------------------------------------------*/
/*                                          SPECIAL IDENTITY REPRESENTATION */
ADOLC_DLL_EXPORT double   **myallocI2(int);
ADOLC_DLL_EXPORT void myfreeI2(int, double**);

ADOLC_DLL_EXPORT unsigned int * myalloc1_uint(int);

ADOLC_DLL_EXPORT unsigned long int *  myalloc1_ulong(int);
ADOLC_DLL_EXPORT unsigned long int ** myalloc2_ulong(int, int);


/****************************************************************************/
/*                              INTEGER VARIANT FOR BIT PATTERN PROPAGATION */

ADOLC_DLL_EXPORT void myfree1_uint(unsigned int*);

ADOLC_DLL_EXPORT void myfree1_ulong(unsigned long int *);
ADOLC_DLL_EXPORT void myfree2_ulong(unsigned long int **);

END_C_DECLS

/****************************************************************************/
/*                                                       Now the C++ THINGS */
#if defined(__cplusplus)

/*--------------------------------------------------------------------------*/
/*                                              MEMORY MANAGEMENT UTILITIES */
inline double   * myalloc(int n) {
    return myalloc1(n);
}
inline double  ** myalloc(int m, int n) {
    return myalloc2(m,n);
}
inline double *** myalloc(int m, int n, int p) {
    return myalloc3(m,n,p);
}

inline void myfree(double   *A) {
    myfree1(A);
}
inline void myfree(double  **A) {
    myfree2(A);
}
inline void myfree(double ***A) {
    myfree3(A);
}

#endif

/****************************************************************************/
#endif
