/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     oplate.h
 Revision: $Id$
 Contents: Numeric values for the various opcodes used by ADOL-C.
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
           
----------------------------------------------------------------------------*/

#if !defined(ADOLC_OPLATE_P_H)
#define ADOLC_OPLATE_P_H 1

#include <common.h>

/****************************************************************************/
/* opcodes */

enum OPCODES {
  death_not,
  assign_ind,
  assign_dep,
  assign_a,
  assign_d,
  eq_plus_d,
  eq_plus_a,
  eq_min_d,
  eq_min_a,
  eq_mult_d,
  eq_mult_a,
  plus_a_a,
  plus_d_a,
  min_a_a,
  min_d_a,
  mult_a_a,
  mult_d_a,
  div_a_a,
  div_d_a,
  exp_op,
  cos_op,
  sin_op,
  atan_op,
  log_op,
  pow_op,
  asin_op,
  acos_op,
  sqrt_op,
  asinh_op,
  acosh_op,
  atanh_op,
  gen_quad,
  end_of_tape,
  start_of_tape,
  end_of_op,
  end_of_int,
  end_of_val,
  cond_assign,
  cond_assign_s,
  take_stock_op,
  assign_d_one,
  assign_d_zero,
  incr_a,
  decr_a,
  neg_sign_a,
  pos_sign_a,
  min_op,
  abs_val,
  eq_zero,
  neq_zero,
  le_zero,
  gt_zero,
  ge_zero,
  lt_zero,
  eq_plus_prod,
  eq_min_prod,
  erf_op,
  ceil_op,
  floor_op,
  ext_diff,
  ignore_me
};

/****************************************************************************/
#endif
