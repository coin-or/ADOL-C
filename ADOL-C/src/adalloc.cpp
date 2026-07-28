/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adalloc.cpp
 Revision: $Id$
 Contents: C allocation of arrays of doubles in several dimensions

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adalloc.h>
#include <adolc/adolcerror.h>
#include <adolc/dvlparms.h>

#if !defined(ADOLC_NO_MALLOC)
#define ADOLC_CALLOC(n, m) calloc(n, m)
#else
#define ADOLC_CALLOC(n, m) rpl_calloc(n, m)
#endif
#if !defined(ADOLC_NO_MALLOC)
#define ADOLC_MALLOC(n, m) calloc(n, m)
#else
#define ADOLC_MALLOC(n, m) rpl_calloc(n, m)
#endif

BEGIN_C_DECLS

/****************************************************************************/
/*                                              MEMORY MANAGEMENT UTILITIES */
/*--------------------------------------------------------------------------*/
char *populate_dpp(double ***const pointer, char *const memory, size_t n,
                   size_t m) {
  char *tmp;
  double **tmp1;
  double *tmp2;
  tmp = (char *)memory;
  tmp1 = (double **)memory;
  *pointer = tmp1;
  tmp = (char *)(tmp1 + n);
  tmp2 = (double *)tmp;
  for (size_t i = 0; i < n; i++) {
    (*pointer)[i] = tmp2;
    tmp2 += m;
  }
  tmp = (char *)tmp2;
  return tmp;
}
/*--------------------------------------------------------------------------*/
char *populate_dppp(double ****const pointer, char *const memory, size_t n,
                    size_t m, size_t p) {
  char *tmp;
  double ***tmp1;
  double **tmp2;
  double *tmp3;
  tmp = (char *)memory;
  tmp1 = (double ***)memory;
  *pointer = tmp1;
  tmp = (char *)(tmp1 + n);
  tmp2 = (double **)tmp;
  for (size_t i = 0; i < n; i++) {
    (*pointer)[i] = tmp2;
    tmp2 += m;
  }
  tmp = (char *)tmp2;
  tmp3 = (double *)tmp;
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < m; j++) {
      (*pointer)[i][j] = tmp3;
      tmp3 += p;
    }
  tmp = (char *)tmp3;
  return tmp;
}
/*--------------------------------------------------------------------------*/
char *populate_dppp_nodata(double ****const pointer, char *const memory,
                           size_t n, size_t m) {

  char *tmp;
  double ***tmp1;
  double **tmp2;
  tmp = (char *)memory;
  tmp1 = (double ***)memory;
  *pointer = tmp1;
  tmp = (char *)(tmp1 + n);
  tmp2 = (double **)tmp;
  for (size_t i = 0; i < n; i++) {
    (*pointer)[i] = tmp2;
    tmp2 += m;
  }
  tmp = (char *)tmp2;
  return tmp;
}

END_C_DECLS