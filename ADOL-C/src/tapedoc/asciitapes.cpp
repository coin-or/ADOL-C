/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     tapedoc/ascii2tape.cpp
 Revision: $Id$
 Contents: Routine ascii2trace(..) converts an ascii description of the trace
           to a real trace in ADOL-C core or disk

 Copyright (c) Kshitij Kulshreshtha


 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#include <adolc/internal/common.h>
#include <adolc/tapedoc/asciitapes.h>

#include "oplate.h"
#include "taping_p.h"
#include "dvlparms.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <limits>
#include <unordered_map>
#include <forward_list>
#include <boost/pool/pool_alloc.hpp>
#include <boost/regex.hpp>

static const std::unordered_map<std::string, enum OPCODES> opcodes = 
{
  { "death_not", death_not },
  { "assign_ind", assign_ind },
  { "assign_dep", assign_dep },
  { "assign_a", assign_a },
  { "assign_d", assign_d },
  { "eq_plus_d", eq_plus_d },
  { "eq_plus_a", eq_plus_a },
  { "eq_min_d", eq_min_d },
  { "eq_min_a", eq_min_a },
  { "eq_mult_d", eq_mult_d },
  { "eq_mult_a", eq_mult_a },
  { "plus_a_a", plus_a_a },
  { "plus_d_a", plus_d_a },
  { "min_a_a", min_a_a },
  { "min_d_a", min_d_a },
  { "mult_a_a", mult_a_a },
  { "mult_d_a", mult_d_a },
  { "div_a_a", div_a_a },
  { "div_d_a", div_d_a },
  { "exp_op", exp_op },
  { "cos_op", cos_op },
  { "sin_op", sin_op },
  { "atan_op", atan_op },
  { "log_op", log_op },
  { "pow_op", pow_op },
  { "asin_op", asin_op },
  { "acos_op", acos_op },
  { "sqrt_op", sqrt_op },
  { "asinh_op", asinh_op },
  { "acosh_op", acosh_op },
  { "atanh_op", atanh_op },
  { "gen_quad", gen_quad },
  { "end_of_tape", end_of_tape },
  { "start_of_tape", start_of_tape },
  { "end_of_op", end_of_op },
  { "end_of_int", end_of_int },
  { "end_of_val", end_of_val },
  { "cond_assign", cond_assign },
  { "cond_assign_s", cond_assign_s },
  { "take_stock_op", take_stock_op },
  { "assign_d_one", assign_d_one },
  { "assign_d_zero", assign_d_zero },
  { "incr_a", incr_a },
  { "decr_a", decr_a },
  { "neg_sign_a", neg_sign_a },
  { "pos_sign_a", pos_sign_a },
  { "min_op", min_op },
  { "abs_val", abs_val },
  { "eq_zero", eq_zero },
  { "neq_zero", neq_zero },
  { "le_zero", le_zero },
  { "gt_zero", gt_zero },
  { "ge_zero", ge_zero },
  { "lt_zero", lt_zero },
  { "eq_plus_prod", eq_plus_prod },
  { "eq_min_prod", eq_min_prod },
  { "erf_op", erf_op },
  { "ceil_op", ceil_op },
  { "floor_op", floor_op },
  { "ext_diff", ext_diff },
  { "ext_diff_iArr", ext_diff_iArr },
  { "ignore_me", ignore_me },
  { "ext_diff_v", ext_diff_v2 },
  { "cond_eq_assign", cond_eq_assign },
  { "cond_eq_assign_s", cond_eq_assign_s },
  { "set_numparam", set_numparam },
  { "subscript", subscript },
  { "subscript_ref", subscript_ref },
  { "ref_assign_d_zero", ref_assign_d_zero },
  { "ref_assign_d_one", ref_assign_d_one },
  { "ref_assign_d", ref_assign_d },
  { "ref_assign_a", ref_assign_a },
  { "ref_assign_ind", ref_assign_ind },
  { "ref_incr_a", ref_incr_a },
  { "ref_decr_a", ref_decr_a },
  { "ref_eq_plus_d", ref_eq_plus_d },
  { "ref_eq_min_d", ref_eq_min_d },
  { "ref_eq_plus_a", ref_eq_plus_a },
  { "ref_eq_min_a", ref_eq_min_a },
  { "ref_eq_mult_d", ref_eq_mult_d },
  { "ref_eq_mult_a", ref_eq_mult_a },
  { "ref_copyout", ref_copyout },
  { "ref_cond_assign", ref_cond_assign },
  { "ref_cond_assign_s", ref_cond_assign_s },
  { "assign_p", assign_p },
  { "eq_plus_p", eq_plus_p },
  { "eq_min_p", eq_min_p },
  { "eq_mult_p", eq_mult_p },
  { "ref_assign_p", ref_assign_p },
  { "ref_eq_plus_p", ref_eq_plus_p },
  { "ref_eq_min_p", ref_eq_min_p },
  { "ref_eq_mult_p", ref_eq_mult_p },
  { "plus_a_p", plus_a_p },
  { "min_a_p", min_a_p },
  { "mult_a_p", mult_a_p },
  { "div_p_a", div_p_a },
  { "pow_op_p", pow_op_p },
  { "neg_sign_p", neg_sign_p },
  { "recipr_p", recipr_p },
  { "vec_copy", vec_copy },
  { "vec_dot", vec_dot },
  { "vec_axpy", vec_axpy },
  { "ref_cond_eq_assign", ref_cond_eq_assign },
  { "ref_cond_eq_assign_s", ref_cond_eq_assign_s },
  { "eq_a_p", eq_a_p },
  { "neq_a_p", neq_a_p },
  { "le_a_p", le_a_p },
  { "gt_a_p", gt_a_p },
  { "ge_a_p", ge_a_p },
  { "lt_a_p", lt_a_p },
  { "eq_a_a", eq_a_a },
  { "neq_a_a", neq_a_a },
  { "le_a_a", le_a_a },
  { "gt_a_a", gt_a_a },
  { "ge_a_a", ge_a_a },
  { "lt_a_a", lt_a_a },
  { "ampi_send", ampi_send },
  { "ampi_recv", ampi_recv },
  { "ampi_isend", ampi_isend },
  { "ampi_irecv", ampi_irecv },
  { "ampi_wait", ampi_wait },
  { "ampi_barrier", ampi_barrier },
  { "ampi_gather", ampi_gather },
  { "ampi_scatter", ampi_scatter },
  { "ampi_allgather", ampi_allgather },
  { "ampi_gatherv", ampi_gatherv },
  { "ampi_scatterv", ampi_scatterv },
  { "ampi_allgatherv", ampi_allgatherv },
  { "ampi_bcast", ampi_bcast },
  { "ampi_reduce", ampi_reduce },
  { "ampi_allreduce", ampi_allreduce }
};

static const std::unordered_map<unsigned char, std::string> opnames = {
  { death_not, "death_not" },
  { assign_ind, "assign_ind" },
  { assign_dep, "assign_dep" },
  { assign_a, "assign_a" },
  { assign_d, "assign_d" },
  { eq_plus_d, "eq_plus_d" },
  { eq_plus_a, "eq_plus_a" },
  { eq_min_d, "eq_min_d" },
  { eq_min_a, "eq_min_a" },
  { eq_mult_d, "eq_mult_d" },
  { eq_mult_a, "eq_mult_a" },
  { plus_a_a, "plus_a_a" },
  { plus_d_a, "plus_d_a" },
  { min_a_a, "min_a_a" },
  { min_d_a, "min_d_a" },
  { mult_a_a, "mult_a_a" },
  { mult_d_a, "mult_d_a" },
  { div_a_a, "div_a_a" },
  { div_d_a, "div_d_a" },
  { exp_op, "exp_op" },
  { cos_op, "cos_op" },
  { sin_op, "sin_op" },
  { atan_op, "atan_op" },
  { log_op, "log_op" },
  { pow_op, "pow_op" },
  { asin_op, "asin_op" },
  { acos_op, "acos_op" },
  { sqrt_op, "sqrt_op" },
  { asinh_op, "asinh_op" },
  { acosh_op, "acosh_op" },
  { atanh_op, "atanh_op" },
  { gen_quad, "gen_quad" },
  { end_of_tape, "end_of_tape" },
  { start_of_tape, "start_of_tape" },
  { end_of_op, "end_of_op" },
  { end_of_int, "end_of_int" },
  { end_of_val, "end_of_val" },
  { cond_assign, "cond_assign" },
  { cond_assign_s, "cond_assign_s" },
  { take_stock_op, "take_stock_op" },
  { assign_d_one, "assign_d_one" },
  { assign_d_zero, "assign_d_zero" },
  { incr_a, "incr_a" },
  { decr_a, "decr_a" },
  { neg_sign_a, "neg_sign_a" },
  { pos_sign_a, "pos_sign_a" },
  { min_op, "min_op" },
  { abs_val, "abs_val" },
  { eq_zero, "eq_zero" },
  { neq_zero, "neq_zero" },
  { le_zero, "le_zero" },
  { gt_zero, "gt_zero" },
  { ge_zero, "ge_zero" },
  { lt_zero, "lt_zero" },
  { eq_plus_prod, "eq_plus_prod" },
  { eq_min_prod, "eq_min_prod" },
  { erf_op, "erf_op" },
  { ceil_op, "ceil_op" },
  { floor_op, "floor_op" },
  { ext_diff, "ext_diff" },
  { ext_diff_iArr, "ext_diff_iArr" },
  { ignore_me, "ignore_me" },
  { ext_diff_v2, "ext_diff_v" },
  { cond_eq_assign, "cond_eq_assign" },
  { cond_eq_assign_s, "cond_eq_assign_s" },
  { set_numparam, "set_numparam" },
  { subscript, "subscript" },
  { subscript_ref, "subscript_ref" },
  { ref_assign_d_zero, "ref_assign_d_zero" },
  { ref_assign_d_one, "ref_assign_d_one" },
  { ref_assign_d, "ref_assign_d" },
  { ref_assign_a, "ref_assign_a" },
  { ref_assign_ind, "ref_assign_ind" },
  { ref_incr_a, "ref_incr_a" },
  { ref_decr_a, "ref_decr_a" },
  { ref_eq_plus_d, "ref_eq_plus_d" },
  { ref_eq_min_d, "ref_eq_min_d" },
  { ref_eq_plus_a, "ref_eq_plus_a" },
  { ref_eq_min_a, "ref_eq_min_a" },
  { ref_eq_mult_d, "ref_eq_mult_d" },
  { ref_eq_mult_a, "ref_eq_mult_a" },
  { ref_copyout, "ref_copyout" },
  { ref_cond_assign, "ref_cond_assign" },
  { ref_cond_assign_s, "ref_cond_assign_s" },
  { assign_p, "assign_p" },
  { eq_plus_p, "eq_plus_p" },
  { eq_min_p, "eq_min_p" },
  { eq_mult_p, "eq_mult_p" },
  { ref_assign_p, "ref_assign_p" },
  { ref_eq_plus_p, "ref_eq_plus_p" },
  { ref_eq_min_p, "ref_eq_min_p" },
  { ref_eq_mult_p, "ref_eq_mult_p" },
  { plus_a_p, "plus_a_p" },
  { min_a_p, "min_a_p" },
  { mult_a_p, "mult_a_p" },
  { div_p_a, "div_p_a" },
  { pow_op_p, "pow_op_p" },
  { neg_sign_p, "neg_sign_p" },
  { recipr_p, "recipr_p" },
  { vec_copy, "vec_copy" },
  { vec_dot, "vec_dot" },
  { vec_axpy, "vec_axpy" },
  { ref_cond_eq_assign, "ref_cond_eq_assign" },
  { ref_cond_eq_assign_s, "ref_cond_eq_assign_s" },
  { eq_a_p, "eq_a_p" },
  { neq_a_p, "neq_a_p" },
  { le_a_p, "le_a_p" },
  { gt_a_p, "gt_a_p" },
  { ge_a_p, "ge_a_p" },
  { lt_a_p, "lt_a_p" },
  { eq_a_a, "eq_a_a" },
  { neq_a_a, "neq_a_a" },
  { le_a_a, "le_a_a" },
  { gt_a_a, "gt_a_a" },
  { ge_a_a, "ge_a_a" },
  { lt_a_a, "lt_a_a" },
  { ampi_send, "ampi_send" },
  { ampi_recv, "ampi_recv" },
  { ampi_isend, "ampi_isend" },
  { ampi_irecv, "ampi_irecv" },
  { ampi_wait, "ampi_wait" },
  { ampi_barrier, "ampi_barrier" },
  { ampi_gather, "ampi_gather" },
  { ampi_scatter, "ampi_scatter" },
  { ampi_allgather, "ampi_allgather" },
  { ampi_gatherv, "ampi_gatherv" },
  { ampi_scatterv, "ampi_scatterv" },
  { ampi_allgatherv, "ampi_allgatherv" },
  { ampi_bcast, "ampi_bcast" },
  { ampi_reduce, "ampi_reduce" },
  { ampi_allreduce, "ampi_allreduce" }
};

typedef std::unordered_map<unsigned char, size_t> requiredargs_t;

static const requiredargs_t num_req_loc = {
    { death_not, 2 }, 
    { assign_ind, 1 },
    { assign_dep, 1 },
    { assign_a, 2 },
    { assign_d, 1 },
    { eq_plus_d, 1 },
    { eq_plus_a, 2 },
    { eq_min_d, 1 },
    { eq_min_a, 2 },
    { eq_mult_d, 1 },
    { eq_mult_a, 2 },
    { plus_a_a, 3 },
    { plus_d_a, 2 },
    { min_a_a, 3 },
    { min_d_a, 2 },
    { mult_a_a, 3 },
    { mult_d_a, 2 },
    { div_a_a, 3 },
    { div_d_a, 2 },
    { exp_op, 2 },
    { cos_op, 3 },
    { sin_op, 3 },
    { atan_op, 3 },
    { log_op, 2 },
    { pow_op, 2 },
    { asin_op, 3 },
    { acos_op, 3 },
    { sqrt_op, 2 },
    { asinh_op, 3 },
    { acosh_op, 3 },
    { atanh_op, 3 },
    { gen_quad, 3 },
    { end_of_tape, 0 },
    { start_of_tape, 0 },
    { end_of_op, 0 },
    { end_of_int, 0 },
    { end_of_val, 0 },
    { cond_assign, 4 },
    { cond_assign_s, 3 },
    { take_stock_op, 2 },
    { assign_d_one, 1 },
    { assign_d_zero, 1 },
    { incr_a, 1 },
    { decr_a, 1 },
    { neg_sign_a, 2 },
    { pos_sign_a, 2 },
    { min_op, 3 },
    { abs_val, 2 },
    { eq_zero, 1 },
    { neq_zero, 1 },
    { le_zero, 1 },
    { gt_zero, 1 },
    { ge_zero, 1 },
    { lt_zero, 1 },
    { eq_plus_prod, 3 },
    { eq_min_prod, 3 },
    { erf_op, 3 },
    { ceil_op, 2 },
    { floor_op, 2 },
    { ext_diff, 6 },
    { ext_diff_iArr, 65536 },
    { ignore_me, 0 },
    { ext_diff_v2, 65536 },
    { cond_eq_assign, 4 },
    { cond_eq_assign_s, 3 },
    { set_numparam, 1 },
    { subscript, 3 },
    { subscript_ref, 3 },
    { ref_assign_d_zero, 1 },
    { ref_assign_d_one, 1 },
    { ref_assign_d, 1 },
    { ref_assign_a, 2 },
    { ref_assign_ind, 1 },
    { ref_incr_a, 1 },
    { ref_decr_a, 1 },
    { ref_eq_plus_d, 1 },
    { ref_eq_min_d, 1 },
    { ref_eq_plus_a, 2 },
    { ref_eq_min_a, 3 },
    { ref_eq_mult_d, 1 },
    { ref_eq_mult_a, 2 },
    { ref_copyout, 2 },
    { ref_cond_assign, 4 },
    { ref_cond_assign_s, 3 },
    { assign_p, 2 },
    { eq_plus_p, 2 },
    { eq_min_p, 2 },
    { eq_mult_p, 2 },
    { ref_assign_p, 2 },
    { ref_eq_plus_p, 2 },
    { ref_eq_min_p, 2 },
    { ref_eq_mult_p, 2 },
    { plus_a_p, 3 },
    { min_a_p, 3 },
    { mult_a_p, 3 },
    { div_p_a, 3 },
    { pow_op_p, 3 },
    { neg_sign_p, 2 },
    { recipr_p, 2 },
    { vec_copy, 3 },
    { vec_dot, 4 },
    { vec_axpy, 5 },
    { ref_cond_eq_assign, 4 },
    { ref_cond_eq_assign_s, 3 },
    { eq_a_p, 3 },
    { neq_a_p, 3 },
    { le_a_p, 3 },
    { gt_a_p, 3 },
    { ge_a_p, 3 },
    { lt_a_p, 3 },
    { eq_a_a, 3 },
    { neq_a_a, 3 },
    { le_a_a, 3 },
    { gt_a_a, 3 },
    { ge_a_a, 3 },
    { lt_a_a, 3 },
    { ampi_send, 1 },
    { ampi_recv, 1 },
    { ampi_isend, 0 },
    { ampi_irecv, 0 },
    { ampi_wait, 0 },
    { ampi_barrier, 0 },
    { ampi_gather, 2 },
    { ampi_scatter, 2 },
    { ampi_allgather, 2 },
    { ampi_gatherv, 2 },
    { ampi_scatterv, 2 },
    { ampi_allgatherv, 2 },
    { ampi_bcast, 1 },
    { ampi_reduce, 2 },
    { ampi_allreduce, 0 }
};

static const requiredargs_t num_req_val = {
    {  death_not, 0 }, 
    { assign_ind, 0 },
    { assign_dep, 0 },
    { assign_a, 0 },
    { assign_d, 1 },
    { eq_plus_d, 1 },
    { eq_plus_a, 0 },
    { eq_min_d, 1 },
    { eq_min_a, 0 },
    { eq_mult_d, 1 },
    { eq_mult_a, 0 },
    { plus_a_a, 0 },
    { plus_d_a, 1 },
    { min_a_a, 0 },
    { min_d_a, 1 },
    { mult_a_a, 0 },
    { mult_d_a, 1 },
    { div_a_a, 0 },
    { div_d_a, 1 },
    { exp_op, 0 },
    { cos_op, 0 },
    { sin_op, 0 },
    { atan_op, 0 },
    { log_op, 0 },
    { pow_op, 1 },
    { asin_op, 0 },
    { acos_op, 0 },
    { sqrt_op, 0 },
    { asinh_op, 0 },
    { acosh_op, 0 },
    { atanh_op, 0 },
    { gen_quad, 2 },
    { end_of_tape, 0 },
    { start_of_tape, 0 },
    { end_of_op, 0 },
    { end_of_int, 0 },
    { end_of_val, 0 },
    { cond_assign, 1 },
    { cond_assign_s, 1 },
    { take_stock_op, 65536 },
    { assign_d_one, 0 },
    { assign_d_zero, 0 },
    { incr_a, 0 },
    { decr_a, 0 },
    { neg_sign_a, 0 },
    { pos_sign_a, 0 },
    { min_op, 1 },
    { abs_val, 1 },
    { eq_zero, 0 },
    { neq_zero, 0 },
    { le_zero, 0 },
    { gt_zero, 0 },
    { ge_zero, 0 },
    { lt_zero, 0 },
    { eq_plus_prod, 0 },
    { eq_min_prod, 0 },
    { erf_op, 0 },
    { ceil_op, 1 },
    { floor_op, 1 },
    { ext_diff, 0 },
    { ext_diff_iArr, 0 },
    { ignore_me, 0 },
    { ext_diff_v2, 0 },
    { cond_eq_assign, 1 },
    { cond_eq_assign_s, 1 },
    { set_numparam, 0 },
    { subscript, 1 },
    { subscript_ref, 1 },
    { ref_assign_d_zero, 0 },
    { ref_assign_d_one, 0 },
    { ref_assign_d, 1 },
    { ref_assign_a, 0 },
    { ref_assign_ind, 0 },
    { ref_incr_a, 0 },
    { ref_decr_a, 0 },
    { ref_eq_plus_d, 1 },
    { ref_eq_min_d, 1 },
    { ref_eq_plus_a, 0 },
    { ref_eq_min_a, 0 },
    { ref_eq_mult_d, 1 },
    { ref_eq_mult_a, 0 },
    { ref_copyout, 0 },
    { ref_cond_assign, 1 },
    { ref_cond_assign_s, 1 },
    { assign_p, 0 },
    { eq_plus_p, 0 },
    { eq_min_p, 0 },
    { eq_mult_p, 0 },
    { ref_assign_p, 0 },
    { ref_eq_plus_p, 0 },
    { ref_eq_min_p, 0 },
    { ref_eq_mult_p, 0 },
    { plus_a_p, 0 },
    { min_a_p, 0 },
    { mult_a_p, 0 },
    { div_p_a, 0 },
    { pow_op_p, 0 },
    { neg_sign_p, 0 },
    { recipr_p, 0 },
    { vec_copy, 0 },
    { vec_dot, 0 },
    { vec_axpy, 0 },
    { ref_cond_eq_assign, 1 },
    { ref_cond_eq_assign_s, 1 },
    { eq_a_p, 1 },
    { neq_a_p, 1 },
    { le_a_p, 1 },
    { gt_a_p, 1 },
    { ge_a_p, 1 },
    { lt_a_p, 1 },
    { eq_a_a, 1 },
    { neq_a_a, 1 },
    { le_a_a, 1 },
    { gt_a_a, 1 },
    { ge_a_a, 1 },
    { lt_a_a, 1 },
    { ampi_send, 0 },
    { ampi_recv, 0 },
    { ampi_isend, 0 },
    { ampi_irecv, 0 },
    { ampi_wait, 0 },
    { ampi_barrier, 0 },
    { ampi_gather, 0 },
    { ampi_scatter, 0 },
    { ampi_allgather, 0 },
    { ampi_gatherv, 0 },
    { ampi_scatterv, 0 },
    { ampi_allgatherv, 0 },
    { ampi_bcast, 0 },
    { ampi_reduce, 0 },
    { ampi_allreduce, 0 }
};

template<typename _Alloc>
static void handle_ops_stats(enum OPCODES operation,
                             std::deque<locint, _Alloc>& locs) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (operation >= ampi_send) {
        fprintf(DIAG_OUT,"ADOL-C error: deserializing AdjoinableMPI operation is not supported\n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    switch (operation) {
        case ref_assign_ind:
        case assign_ind:
            ++ADOLC_CURRENT_TAPE_INFOS.numInds;
            break;
        case assign_dep:
            ++ADOLC_CURRENT_TAPE_INFOS.numDeps;
            break;
        case eq_plus_prod:
            ++ADOLC_CURRENT_TAPE_INFOS.num_eq_prod;
            break;
        case eq_min_prod:
            ++ADOLC_CURRENT_TAPE_INFOS.num_eq_prod;
            break; 
        case vec_axpy:
        {
            locint n = *(++locs.crbegin());
            ADOLC_CURRENT_TAPE_INFOS.num_eq_prod += 2*n-1;
        }
            break;
        case vec_dot:
        {
            locint n = *(++locs.crbegin());
            ADOLC_CURRENT_TAPE_INFOS.num_eq_prod += 2*n;
        }
            break;
        case abs_val:
            if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
                ++ADOLC_CURRENT_TAPE_INFOS.numSwitches;
            break;
        case assign_p:
        case recipr_p:
        case neg_sign_p:
        case eq_plus_p:
        case eq_min_p:
        case eq_mult_p:
        case ref_eq_plus_p:
        case ref_eq_min_p:
        case ref_eq_mult_p:
        {
            locint n = *locs.cbegin();
            if (ADOLC_GLOBAL_TAPE_VARS.numparam <= n) 
                ADOLC_GLOBAL_TAPE_VARS.numparam = n+1;
        }
            break;
        case plus_a_p:
        case min_a_p:
        case mult_a_p:
        case pow_op_p:
        case neq_a_p:
        case eq_a_p:
        case le_a_p:
        case ge_a_p:
        case lt_a_p:
        case gt_a_p:
        case ref_assign_p:
        case set_numparam:
        {
            locint n = *(++locs.cbegin());
            if (ADOLC_GLOBAL_TAPE_VARS.numparam <= n) 
                ADOLC_GLOBAL_TAPE_VARS.numparam = n+1;
        }
            break;
        default:
            break;
    }
}

BEGIN_C_DECLS

static locint maxloc = 4;

static void get_ascii_trace_elements(const std::string& instr) {
    ADOLC_OPENMP_THREAD_NUMBER;

    std::string oppat = "op:([_a-z]+)",
        locpat = "loc:([0-9]+)",
        valpat = "val:([+-]?[0-9]+\\.?[0-9]*(e[+-][0-9]+)?)";
    boost::regex opexp(oppat,boost::regex::perl|boost::regex::icase),
        locexp(locpat,boost::regex::perl|boost::regex::icase),
        valexp(valpat,boost::regex::perl|boost::regex::icase);
    boost::sregex_iterator iend;
    boost::sregex_iterator opa(instr.begin(), instr.end(), opexp, boost::match_default),
        loca(instr.begin(), instr.end(), locexp, boost::match_default),
        vala(instr.begin(), instr.end(), valexp, boost::match_default);
    std::deque<locint,boost::fast_pool_allocator<locint> > locs;
    size_t opctr = 0, valctr = 0, locctr = 0;

    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (!ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
        fprintf(DIAG_OUT, "ADOL-C error: get_ascii_trace_elements() called without starting a trace\n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }

    while(opa != iend) {
        enum OPCODES oper = opcodes.at((*opa)[1].str());
        if (oper == ext_diff_iArr) {
            locint iarrlen = std::strtoul((*loca)[1].str().c_str(),NULL,0);
            locs.push_back(iarrlen);
            ++loca;
            put_op_reserve(oper,iarrlen+2);
            ADOLC_PUT_LOCINT(iarrlen);
        } else if (oper == ext_diff_v2) {
            locint idx = std::strtoul((*loca)[1].str().c_str(),NULL,0); 
            locs.push_back(idx);
            ++loca;
            locint iarrlen = std::strtoul((*loca)[1].str().c_str(),NULL,0);
            locs.push_back(iarrlen);
            ++loca;
            auto nextlocat = loca;
            for (locint i = 0; i < iarrlen + 1; i++) {
                if (nextlocat == iend) {
                    fprintf(DIAG_OUT, "ADOL-C error: not enough locations given in ext_diff_v");
                    adolc_exit(-1,"",__func__,__FILE__,__LINE__);
                }
                ++nextlocat;
            }
            locint nin = std::strtoul((*nextlocat)[1].str().c_str(),NULL,0);
            ++nextlocat;
            locint nout = std::strtoul((*nextlocat)[1].str().c_str(),NULL,0);
            put_op_reserve(oper, 2*(nin+nout)+iarrlen);
            ADOLC_PUT_LOCINT(idx);
            ADOLC_PUT_LOCINT(iarrlen);
        } if (oper == take_stock_op) {
            // we really shouldn't have take_stock_op in our trace since
            // trace_on() already writes one, so even if some externally
            // written trace contains it, we'll make sure we use assign_d type
            // operations instead.
            locint start = std::strtoul((*loca)[1].str().c_str(),NULL,0);
            ++loca;
            locint number = std::strtoul((*loca)[1].str().c_str(),NULL,0);
            while (maxloc < start+number) maxloc *= 2;
            for (locint i=start; i < start+number; i++) {
                if (vala == iend) {
                    fprintf(DIAG_OUT, "ADOL-C error: not enough values given in take_stock_op");
                    adolc_exit(-1,"",__func__,__FILE__,__LINE__);
                }
                double val = strtod((*vala)[1].str().c_str(),NULL);
                ++vala;
                if (val == 0.0) {
                    put_op(assign_d_zero);
                    ADOLC_PUT_LOCINT(i);
                } else if (val == 1.0) {
                    put_op(assign_d_one);
                    ADOLC_PUT_LOCINT(i);
                } else {
                    put_op(assign_d);
                    ADOLC_PUT_LOCINT(i);
                    ADOLC_PUT_VAL(val);
                }
            }
        } if (oper == death_not) {
            // we count maxlocs ourselves and trace_off() writes a death_not
            // so here we should only check if the given value is bigger than
            // the maxloc we've got
            locint idx = std::strtoul((*loca)[1].str().c_str(),NULL,0); 
            locs.push_back(idx);
            ++loca;
            ++locctr;
            idx = std::strtoul((*loca)[1].str().c_str(),NULL,0);
            locs.push_back(idx);
            ++loca;
            ++locctr;
            if (idx > maxloc) maxloc *= 2;
        } if (oper == set_numparam) {
            locint idx = std::strtoul((*loca)[1].str().c_str(),NULL,0);
            locs.push_back(idx);
            ++loca;
            ++locctr;
        } else 
            put_op(oper);
        while (loca != iend) {
            locint loc = std::strtoul((*loca)[1].str().c_str(),NULL,0);
            while (maxloc < loc) maxloc *= 2;
            locs.push_back(loc);
            ADOLC_PUT_LOCINT(loc);
            ++loca;
            ++locctr;
        }
        if (oper != ext_diff_iArr && oper != ext_diff_v2 && locctr > num_req_loc.at(oper)) {
            std::cout << "something went wrong, there are " << locctr << "locs in one tag for " << (*opa)[1].str() << "\n";
            adolc_exit(-1,"",__func__,__FILE__,__LINE__);
        }
        if (oper == take_stock_op && vala != iend) {
            std::cout << "something went wrong, there are too many values in one tag for " << (*opa)[1].str() << "\n";
            adolc_exit(-1,"",__func__,__FILE__,__LINE__);
        }
        while (vala != iend) {
            double val = std::strtod((*vala)[1].str().c_str(),NULL);
            ADOLC_PUT_VAL(val);
            ++vala;
            ++valctr;
        }
        if (oper != take_stock_op && valctr > num_req_val.at(oper)) {
            std::cout << "something went wrong, there are " << valctr << "vales in one tag for " << (*opa)[1].str() << "\n";
            adolc_exit(-1,"",__func__,__FILE__,__LINE__);
        }
        handle_ops_stats(oper,locs);
        ++opa;
        ++opctr;
        locs.clear();
        locctr = 0;
        valctr = 0;
    }
    if (opctr > 1) {
        std::cout << "something went wrong, there are " << opctr << "ops in one tag\n";
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
}
                        
void read_ascii_trace(const char*const fname, short tag) {
    char buf[4194304];
    std::ifstream is;

    std::string pattern = "\\{\\s*op:[_a-z]+(\\s+loc:[0-9]+)+\\s*(\\s*val:[+-]?[0-9]+\\.?[0-9]*(e[+-][0-9]+)?)*\\s*\\}";

    is.open(fname);
    if (! is.is_open() ) {
        fprintf(DIAG_OUT, "ADOL-C error: cannot open file %s !\n", fname);
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    trace_on(tag);

    // regular expression we're looking for
    boost::regex outer_expr(pattern,boost::regex::perl|boost::regex::icase);
    // buffer we'll be searching in:
   // saved position of end of partial match:
    const char* next_pos = buf + sizeof(buf);
    // flag to indicate whether there is more input to come:
    bool have_more = true;

    while(have_more) {
        // how much do we copy forward from last try:
        size_t leftover = (buf + sizeof(buf)) - next_pos;
        // and how much is left to fill:
        size_t size = next_pos - buf;
        // copy forward whatever we have left:
        std::memmove(buf, next_pos, leftover);
        // fill the rest from the stream:
        is.read(buf + leftover, size);
        size_t read = is.gcount();
        // check to see if we've run out of text:
        have_more = read == size;
        // reset next_pos:
        next_pos = buf + sizeof(buf);
        // and then iterate:
        boost::cregex_iterator a(
            buf,
            buf + read + leftover,
            outer_expr,
            boost::match_default | boost::match_partial | boost::match_single_line);
        boost::cregex_iterator b;
        
        while(a != b) {
            if((*a)[0].matched == false) {
                // Partial match, save position and break:
                next_pos = (*a)[0].first;
                break;
            }
            else {
                // full match:
                get_ascii_trace_elements(a->str());
            }
            
            // move to next match:
            ++a;
        }
    }
    ADOLC_GLOBAL_TAPE_VARS.storeSize = maxloc;
    if (ADOLC_GLOBAL_TAPE_VARS.pStore != NULL) 
        delete[] ADOLC_GLOBAL_TAPE_VARS.pStore;
    ADOLC_GLOBAL_TAPE_VARS.pStore = new double[ADOLC_GLOBAL_TAPE_VARS.numparam];
    memset(ADOLC_GLOBAL_TAPE_VARS.pStore,0,ADOLC_GLOBAL_TAPE_VARS.numparam*sizeof(double));
    trace_off();
    fprintf(DIAG_OUT,"ADOL-C Warning: reading ascii trace creates no taylor stack\n"
        "Remember to run forward mode with correct setup first.\n");
}

void write_ascii_trace(const char *const fname, short tag) {
    std::ofstream file;
    file.open(fname);
    if (! file.is_open() ) {
        fprintf(DIAG_OUT, "ADOL-C error: cannot open file %s !\n", fname);
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    init_for_sweep(tag);

    unsigned char operation=get_op_f();

    while (operation !=end_of_tape) {
        std::ostringstream outstr;
        if (operation >= ampi_send) {
            fprintf(DIAG_OUT,"ADOL-C error: serializing AdjoinableMPI operation is not supported\n");
            adolc_exit(-1,"",__func__,__FILE__,__LINE__);
        }
        switch (operation) {
            case end_of_op:                                          /* end_of_op */
                get_op_block_f();
                operation=get_op_f();
                /* Skip next operation, it's another end_of_op */
                break;
            case end_of_int:                                        /* end_of_int */
                get_loc_block_f();
                break;
            case end_of_val:                                        /* end_of_val */
               get_val_block_f();
                break;
            case start_of_tape:                                  /* start_of_tape */
            case end_of_tape:                                      /* end_of_tape */
                break;
            case take_stock_op:                                  /* take_stock_op */
            {
                locint size = get_locint_f();
                locint res  = get_locint_f();
                double *d = get_val_v_f(size);
                for (locint ls = 0; ls < size; ls++ )
		    if (d[ls] == 0.0)
			outstr << "{ op:" << opnames.at(assign_d_zero) << " loc:" << res+ls <<  " }\n";
		    else if (d[ls] == 1.0)
			outstr << "{ op:" << opnames.at(assign_d_one) << " loc:" << res+ls <<  " }\n";
		    else
			outstr << "{ op:" << opnames.at(assign_d) << " loc:" << res+ls << " val:" << d[ls] << " }\n";
            }
            break;
            case ext_diff_iArr:
            {
                locint iArrLength = get_locint_f();
                outstr << "{ op:" << opnames.at(operation) << " loc:" << iArrLength;
                for (locint i = 0; i < iArrLength; i++) {
                    locint num = get_locint_f();
                    outstr << " loc:" << num;
                }
                iArrLength = get_locint_f();
                outstr << " loc:" << iArrLength;
                for (locint i = 0; i < 6; i++) {
                    locint num = get_locint_f();
                    outstr << " loc:" << num;
                }
                outstr << " }\n";
            }
            break;
            case ext_diff_v2:
            {
                locint idx = get_locint_f();
                locint iarrlen = get_locint_f();
                outstr << "{ op:" << opnames.at(operation) <<  " loc:" << idx << " loc:" << iarrlen;
                for (locint i = 0 ; i < iarrlen ; i++) {
                    locint num = get_locint_f();
                    outstr << " loc:" << num;
                }
                iarrlen = get_locint_f();
                outstr << " loc:" << iarrlen;
                locint nin = get_locint_f();
                locint nout = get_locint_f();
                outstr << " loc:" << nin << " loc:" << nout;
                for (locint i = 0 ; i < nin ; i++) {
                    locint num1 = get_locint_f();
                    locint num2 = get_locint_f();
                    outstr << " loc:" << num1 << " loc:" << num2;
                }
                for (locint i = 0 ; i < nout ; i++) {
                    locint num1 = get_locint_f();
                    locint num2 = get_locint_f();
                    outstr << " loc:" << num1 << " loc:" << num2;
                }
                nin = get_locint_f();
                nout = get_locint_f();
                outstr << " loc:" << nin << " loc:" << nout;
                outstr << " }\n";
            }
            break;
            default:
            {
                outstr << "{ op:" << opnames.at(operation);
                for (locint i = 0; i < num_req_loc.at(operation); i++) {
                    locint num = get_locint_f();
                    outstr << " loc:" << num;
                }
                for (locint i = 0; i < num_req_val.at(operation); i++) {
                    double val = get_val_f();
                    outstr << " val:" << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1) << val;
                }
                outstr << " }\n";
            }
        }
        file << outstr.str();
        operation=get_op_f();
    }
    end_sweep();
    file.close();
}

END_C_DECLS
