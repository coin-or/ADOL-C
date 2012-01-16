/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts.h
 Revision: $Id$
 Contents: public functions and data types for extern (differentiated)
           functions.
 
 Copyright (c) Andreas Kowarz

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
                     
----------------------------------------------------------------------------*/

#if !defined(ADOLC_EXTERNFCTS_H)
#define ADOLC_EXTERNFCTS_H 1

#include <adolc/common.h>
#include <adolc/adouble.h>

BEGIN_C_DECLS

typedef int (*ADOLC_ext_fct) (int n, double *x, int m, double *y);

/* A variable of this type is created by reg_ext_fct and a pointer to it is
 * returned. Please do not create a variable of this type yourself. The index
 * is likely to be wrong in this case. Use pointers instead. */
typedef struct {
    ADOLC_ext_fct function;

    int (*zos_forward) (int n, double *dp_x,
                        int m, double *dp_y);
    int (*fos_forward) (int n, double *dp_x, double *dp_X,
                        int m, double *dp_y, double *dp_Y);
    int (*fov_forward) (int n, double *dp_x, int p, double **dpp_X,
                        int m, double *dp_y, double **dpp_Y);
    int (*hos_forward) (int n, double *dp_x, int d, double **dpp_X,
                        int m, double *dp_y, double **dpp_Y);
    int (*hov_forward) (int n, double *dp_x, int d, int p, double ***dppp_X,
                        int m, double *dp_y, double ***dppp_Y);

    int (*fos_reverse) (int m, double *dp_U,
                        int n, double *dp_Z);
    int (*fov_reverse) (int m, int p, double **dpp_U,
                        int n, double **dpp_Z);
    int (*hos_reverse) (int m, double *dp_U,
                        int n, int d, double **dpp_Z);
    int (*hov_reverse) (int m, int p, double **dpp_U,
                        int n, int d, double ***dppp_Z,
                        short **spp_nz);

    /* This variables must be set before calling the functions above. */
    double *dp_x;                /* x[n], x0[n]        - forward mode */
    double *dp_X;                /* x1[n]              - forward mode */
    double **dpp_X;              /* X[n][p], X[n][d]   - forward mode */
    double ***dppp_X;            /* X[n][p][d]         - forward mode */
    double *dp_y;                /* y[n], y0[n]        - forward mode */
    double *dp_Y;                /* y1[n]              - forward mode */
    double **dpp_Y;              /* Y[m][p], Y[m][d]   - forward mode */
    double ***dppp_Y;            /* Y[m][p][d]         - forward mode */

    double *dp_U;                /* u[m]               - reverse mode */
    double **dpp_U;              /* U[q][m]            - reverse mode */
    double *dp_Z;                /* z[n]               - reverse mode */
    double **dpp_Z;              /* Z[q][n], Z[n][d+1] - reverse mode */
    double ***dppp_Z;            /* Z[q][n][d+1]       - reverse mode */

    short **spp_nz;              /* nz[q][n]           - reverse mode */

    locint index;                      /* please do not change */
}
ext_diff_fct;

END_C_DECLS

#if defined(__cplusplus)
/****************************************************************************/
/*                                                          This is all C++ */

ADOLC_DLL_EXPORT ext_diff_fct *reg_ext_fct(ADOLC_ext_fct ext_fct);

ADOLC_DLL_EXPORT int call_ext_fct (ext_diff_fct *edfct,
                                   int n, double *xp, adouble *xa,
                                   int m, double *yp, adouble *ya);

#endif /* __CPLUSPLUS */

/****************************************************************************/
#endif /* ADOLC_EXTERNFCTS_H */

