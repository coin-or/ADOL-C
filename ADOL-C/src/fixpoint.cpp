
/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fixpoint.c
 Revision: $Id$
 Contents: all C functions directly accessing at least one of the four tapes
           (operations, locations, constants, value stack)

 Copyright (c) Andreas Kowarz, Sebastian Schlenkrich

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adolc.h>
#include <adolc/dvlparms.h>
#include <adolc/externfcts.h>
#include <adolc/fixpoint.h>

#include <algorithm>
#include <vector>

/*--------------------------------------------------------------------------*/

/* F(x,u,y,dim_x,dim_u) */
/* norm(x,dim_x)        */
struct fpi_data {
  size_t edf_index;
  size_t sub_tape_num;
  double_F double_func;
  adouble_F adouble_func;
  norm_F norm_func;
  norm_deriv_F norm_deriv_func;
  double epsilon;
  double epsilon_deriv;
  size_t N_max;
  size_t N_max_deriv;
};

static std::vector<fpi_data> fpi_stack;

static int iteration(size_t dim_xu, double *xu, size_t dim_x, double *x_fix) {
  double err;
  const fpi_data &current = fpi_stack.back();

  for (size_t i = 0; i < dim_x; ++i)
    x_fix[i] = xu[i];

  for (size_t k = 1; k <= current.N_max; ++k) {
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i];

    current.double_func(xu, xu + dim_x, x_fix, dim_x, dim_xu - dim_x);

    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i] - xu[i];

    err = current.norm_func(xu, dim_x);
    if (err < current.epsilon)
      return k;
  }
  return -1;
}

static int fp_zos_forward(size_t dim_xu, double *xu, size_t dim_x,
                          double *x_fix) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  double err;
  const size_t edf_index = ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index;

  // Find fpi_stack element with index 'edf_index'.
  auto current =
      std::find_if(fpi_stack.begin(), fpi_stack.end(),
                   [&](auto &&v) { return v.edf_index == edf_index; });

  if (current == fpi_stack.end()) {
    fprintf(stderr, "ADOL-C Error! No edf found for fixpoint iteration.\n");
    adolc_exit(-1, "", __func__, __FILE__, __LINE__);
  }

  for (size_t i = 0; i < dim_x; ++i)
    x_fix[i] = xu[i];

  for (size_t k = 1; k <= current->N_max; ++k) {
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i];

    current->double_func(xu, xu + dim_x, x_fix, dim_x, dim_xu - dim_x);

    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i] - xu[i];

    err = current->norm_func(xu, dim_x);
    if (err < current->epsilon)
      return k;
  }
  return -1;
}

static int fp_fos_forward(size_t dim_xu, double *xu, double *xu_dot,
                          size_t dim_x, double *x_fix, double *x_fix_dot) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // Piggy back
  double err, err_deriv;

  const size_t edf_index = ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index;

  // Find fpi_stack element with index 'edf_index'.
  auto current =
      std::find_if(fpi_stack.begin(), fpi_stack.end(),
                   [&](auto &&v) { return v.edf_index == edf_index; });

  if (current == fpi_stack.end()) {
    fprintf(stderr, "ADOL-C Error! No edf found for fixpoint iteration.\n");
    adolc_exit(-1, "", __func__, __FILE__, __LINE__);
  }
  for (size_t k = 1; (k < current->N_max_deriv) || (k < current->N_max); ++k) {
    if (k > 1) {
      for (size_t i = 0; i < dim_x; ++i)
        xu[i] = x_fix[i];

      for (size_t i = 0; i < dim_x; ++i)
        xu_dot[i] = x_fix_dot[i];
    }

    fos_forward(current->sub_tape_num, dim_x, dim_xu, 0, xu, xu_dot, x_fix,
                x_fix_dot);

    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i] - xu[i];

    err = current->norm_func(xu, dim_x);
    for (size_t i = 0; i < dim_x; ++i)
      xu_dot[i] = x_fix_dot[i] - xu_dot[i];

    err_deriv = current->norm_deriv_func(xu_dot, dim_x);
    if ((err < current->epsilon) && (err_deriv < current->epsilon_deriv)) {
      return k;
    }
  }
  return -1;
}

static int fp_fos_reverse(size_t dim_x, double *x_fix_bar, size_t dim_xu,
                          double *xu_bar, double * /*unused*/,
                          double * /*unused*/) {
  // (d x_fix) / (d x_0) = 0 (!)
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  double err;

  const size_t edf_index = ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index;

  // Find fpi_stack element with index 'edf_index'.
  auto current =
      std::find_if(fpi_stack.begin(), fpi_stack.end(),
                   [&](auto &&v) { return v.edf_index == edf_index; });

  if (current == fpi_stack.end()) {
    fprintf(stderr, "ADOL-C Error! No edf found for fixpoint iteration.\n");
    adolc_exit(-1, "", __func__, __FILE__, __LINE__);
  }

  double *U = new double[dim_xu];
  double *xi = new double[dim_x];

  for (size_t k = 1; k < current->N_max_deriv; ++k) {
    for (size_t i = 0; i < dim_x; ++i)
      xi[i] = U[i];

    fos_reverse(current->sub_tape_num, dim_x, dim_xu, xi, U);

    for (size_t i = 0; i < dim_x; ++i)
      U[i] += x_fix_bar[i];

    for (size_t i = 0; i < dim_x; ++i)
      xi[i] = U[i] - xi[i];

    current->norm_deriv_func(xi, dim_x);
    if (err < current->epsilon_deriv) {
      for (size_t i = 0; i < dim_xu - dim_x; ++i) {
        xu_bar[dim_x + i] += U[dim_x + i];
      }

      delete[] xi;
      delete[] U;
      return k;
    }
  }
  for (size_t i = 0; i < dim_xu - dim_x; ++i)
    xu_bar[dim_x + i] += U[dim_x + i];

  delete[] xi;
  delete[] U;
  return -1;
}

ADOLC_DLL_EXPORT int fp_iteration(size_t sub_tape_num, double_F double_func,
                                  adouble_F adouble_func, norm_F norm_func,
                                  norm_deriv_F norm_deriv_func, double epsilon,
                                  double epsilon_deriv, size_t N_max,
                                  size_t N_max_deriv, adouble *x_0, adouble *u,
                                  adouble *x_fix, size_t dim_x, size_t dim_u) {

  // declare extern differentiated function and data
  ext_diff_fct *edf_iteration = reg_ext_fct(&iteration);
  edf_iteration->zos_forward = &fp_zos_forward;
  edf_iteration->fos_forward = &fp_fos_forward;
  edf_iteration->fos_reverse = &fp_fos_reverse;

  // add new fp information
  fpi_stack.emplace_back(fpi_data{
      edf_iteration->index,
      sub_tape_num,
      double_func,
      adouble_func,
      norm_func,
      norm_deriv_func,
      epsilon,
      epsilon_deriv,
      N_max,
      N_max_deriv,
  });

  // ensure that the adoubles are contiguous
  ensureContiguousLocations(dim_x + dim_u + dim_x);
  // put x and u together
  adouble *xu = new adouble[dim_x + dim_u];

  // we use this new variable instead of x_fix to not overwrite the location
  // of x_fix, which is already stored on the tape.
  adouble *x_fix_new = new adouble[dim_x];

  for (size_t i = 0; i < dim_x; ++i)
    xu[i] = x_0[i];

  for (size_t i = 0; i < dim_u; ++i)
    xu[dim_x + i] = u[i];

  const short old_tape_id = ADOLC_CURRENT_TAPE_INFOS.tapeID;
  const size_t old_trace_flag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
  const int k = call_ext_fct(edf_iteration, dim_x + dim_u, xu, dim_x, x_fix);

  // read out x_fix
  for (size_t i = 0; i < dim_x; ++i)
    x_fix_new[i] = x_fix[i];

  // tape near solution
  trace_on(sub_tape_num, 1);
  for (size_t i = 0; i < dim_x; ++i)
    xu[i] <<= x_fix[i].value();

  for (size_t i = 0; i < dim_u; ++i)
    xu[dim_x + i] <<= u[i].value();

  // IMPORTANT: Dont reuse x_fix here. The location of the x_fix's adoubles
  // could change and the old locations are already stored on the tape. This
  // would cause errors
  adouble_func(xu, xu + dim_x, x_fix_new, dim_x, dim_u);

  double dummy_out;
  for (size_t i = 0; i < dim_x; ++i)
    x_fix_new[i] >>= dummy_out;

  trace_off();

  delete[] xu;
  delete[] x_fix_new;
  return k;
}
