/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc_openmp.h
 Revision: $Id$
 Contents: header file for openmp parallel differentiation

 Copyright (c) Andreas Kowarz
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#if !defined(ADOLC_ADOLC_OPENMP_H)
#define ADOLC_ADOLC_OPENMP_H 1


#include <adolc/internal/common.h>

/*--------------------------------------------------------------------------*/
/*                                                             par_jacobian */
/* par_jacobian(tag, m, n, x[n], J[m][n])                                   */
ADOLC_DLL_EXPORT int par_jacobian(short,int,int,const double*,double**);


typedef struct {
        int advances;
        int takeshots;
        int commands;
        int  turn;
        int reps;
        int range;
        int ch[ADOLC_CHECKUP];
        int oldsnaps;
        int oldfine;
} revolve_nums;

struct ThreadContextCl;
extern struct ADOLC_OpenMP_CL {
	struct ThreadContextCl* ctx;
	revolve_nums revolve_numbers;
#ifdef __cplusplus
	struct ADOLC_OpenMP_CL& operator=(struct ADOLC_OpenMP_CL const& in);
#endif
} ADOLC_OpenMP;

#ifdef _OPENMP
#pragma omp threadprivate(ADOLC_OpenMP)
#endif
/*
extern void beginParallel();
extern void endParallel();

extern int ADOLC_parallel_doCopy;

typedef struct ADOLC_OpenMP {
    inline ADOLC_OpenMP() {}
    inline ADOLC_OpenMP(const ADOLC_OpenMP &arg) {
        ADOLC_parallel_doCopy = 1;
        beginParallel();
    }
    inline ~ADOLC_OpenMP() {
        endParallel();
    }
} ADOLC_OpenMP;

typedef struct ADOLC_OpenMP_NC {
    inline ADOLC_OpenMP_NC() {}
    inline ADOLC_OpenMP_NC(const ADOLC_OpenMP_NC &arg) {
        ADOLC_parallel_doCopy = 0;
        beginParallel();
    }
    inline ~ADOLC_OpenMP_NC() {
        endParallel();
    }
} ADOLC_OpenMP_NC;

extern ADOLC_OpenMP ADOLC_OpenMP_Handler;
extern ADOLC_OpenMP_NC ADOLC_OpenMP_Handler_NC;

#define ADOLC_OPENMP firstprivate(ADOLC_OpenMP_Handler)
#define ADOLC_OPENMP_NC firstprivate(ADOLC_OpenMP_Handler_NC)
*/

#endif /* ADOLC_ADOLC_OPENMP_H */

