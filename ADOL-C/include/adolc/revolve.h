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

#ifndef ADOLC_REVOLVE_H
#define ADOLC_REVOLVE_H

#include <adolc/internal/common.h>

struct revolve_nums {
  int advances{0};
  int takeshots{0};
  int commands{0};
  int turn{0};
  int reps{0};
  int range{0};
  int ch[ADOLC_CHECKUP] = {0};
  int oldsnaps{0};
  int oldfine{0};
};

inline revolve_nums revolve_numbers;

inline revolve_nums &get_revolve_numbers() { return revolve_numbers; }

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

int adjustsize(int *steps, int *snaps, int *reps);

enum revolve_action revolve(int *check, int *capo, int *fine, int snaps,
                            int *info);

#endif // ADOLC_REVOLVE_H
