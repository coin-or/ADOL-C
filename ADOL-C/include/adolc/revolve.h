/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     revolve.h
 Revision: $Id$
 Contents: optimal binomial checkpointing adapted for ADOL-C

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

#if !defined(ADOLC_REVOLVE_H)
#define ADOLC_REVOLVE_H 1

#include <adolc/internal/common.h>

BEGIN_C_DECLS

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

#ifndef _OPENMP
    extern revolve_nums revolve_numbers;
#else
#include <omp.h>
    extern revolve_nums *revolve_numbers;
#endif

enum revolve_action {
    revolve_advance,
    revolve_takeshot,
    revolve_restore,
    revolve_firsturn,
    revolve_youturn,
    revolve_terminate,
    revolve_error
};

int maxrange(int ss, int tt);

int adjustsize(int* steps, int* snaps, int* reps);

enum revolve_action revolve(int* check,int* capo,int* fine,int snaps,int* info);

END_C_DECLS

#endif /* ADOLC_REVOLVE_H */

