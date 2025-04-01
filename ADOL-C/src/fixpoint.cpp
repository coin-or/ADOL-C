
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
#include <adolc/adolcerror.h>
#include <adolc/dvlparms.h>
#include <adolc/externfcts.h>
#include <adolc/fixpoint.h>
#include <adolc/valuetape/valuetape.h>
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

static int iteration(short tapeId, size_t dim_xu, double *xu, size_t dim_x,
                     double *x_fix) {
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

static int fp_zos_forward(short tapeId, size_t dim_xu, double *xu, size_t dim_x,
                          double *x_fix) {

  std::shared_ptr<ValueTape> tape = getTape(tapeId);
  double err;
  const size_t edf_index = tape->ext_diff_fct_index();

  // Find fpi_stack element with index 'edf_index'.
  auto current =
      std::find_if(fpi_stack.begin(), fpi_stack.end(),
                   [&](auto &&v) { return v.edf_index == edf_index; });

  if (current == fpi_stack.end())
    fail(ADOLC_ERRORS::ADOLC_FP_NO_EDF, std::source_location::current());

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

static int fp_fos_forward(short tapeId, size_t dim_xu, double *xu,
                          double *xu_dot, size_t dim_x, double *x_fix,
                          double *x_fix_dot) {

  // Piggy back
  double err, err_deriv;
  std::shared_ptr<ValueTape> tape = getTape(tapeId);

  const size_t edf_index = tape->ext_diff_fct_index();

  // Find fpi_stack element with index 'edf_index'.
  auto current =
      std::find_if(fpi_stack.begin(), fpi_stack.end(),
                   [&](auto &&v) { return v.edf_index == edf_index; });

  if (current == fpi_stack.end())
    fail(ADOLC_ERRORS::ADOLC_FP_NO_EDF, std::source_location::current());
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

static int fp_fos_reverse(short tapeId, size_t dim_x, double *x_fix_bar,
                          size_t dim_xu, double *xu_bar, double * /*unused*/,
                          double * /*unused*/) {
  // (d x_fix) / (d x_0) = 0 (!)

  double err;
  std::shared_ptr<ValueTape> tape = getTape(tapeId);
  const size_t edf_index = tape->ext_diff_fct_index();

  // Find fpi_stack element with index 'edf_index'.
  auto current =
      std::find_if(fpi_stack.begin(), fpi_stack.end(),
                   [&](auto &&v) { return v.edf_index == edf_index; });

  if (current == fpi_stack.end())
    fail(ADOLC_ERRORS::ADOLC_FP_NO_EDF, std::source_location::current());

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

int fp_iteration(short tapeId, size_t sub_tape_num, double_F double_func,
                 adouble_F adouble_func, norm_F norm_func,
                 norm_deriv_F norm_deriv_func, double epsilon,
                 double epsilon_deriv, size_t N_max, size_t N_max_deriv,
                 adouble *x_0, adouble *u, adouble *x_fix, size_t dim_x,
                 size_t dim_u) {

  // declare extern differentiated function and data
  ext_diff_fct *edf_iteration = reg_ext_fct(tapeId, sub_tape_num, &iteration);
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

  std::shared_ptr<ValueTape> tape = getTape(tapeId);
  // ensure that the adoubles are contiguous
  tape->ensureContiguousLocations(dim_x + dim_u);
  // to reset old default later
  const short last_default_tape_id = getDefaultTapeIdConst();
  // ensure that the "new" allocates the adoubles for the "tape"
  setDefaultTapeId(tape->tapeId());
  // put x and u together
  adouble *xu = new adouble[dim_x + dim_u];

  std::shared_ptr<ValueTape> sub_tape = getTape(sub_tape_num);
  sub_tape->ensureContiguousLocations(dim_x + dim_u + dim_x);
  // ensure that the "new" allocates the adoubles for the "sub_tape"
  setDefaultTapeId(sub_tape->tapeId());
  // we use this new variable instead of x_fix to not overwrite the location
  // of x_fix, which is already stored on the tape.
  adouble *xu_sub_tape = new adouble[dim_x + dim_u];
  adouble *x_fix_new = new adouble[dim_x];

  for (size_t i = 0; i < dim_x; ++i)
    xu[i] = x_0[i];

  for (size_t i = 0; i < dim_u; ++i)
    xu[dim_x + i] = u[i];

  const int k = call_ext_fct(edf_iteration, dim_x + dim_u, xu, dim_x, x_fix);

  // read out x_fix
  for (size_t i = 0; i < dim_x; ++i)
    x_fix_new[i] = x_fix[i];

  // tape near solution
  trace_on(sub_tape_num, 1);
  for (size_t i = 0; i < dim_x; ++i)
    // xu[i] <<= x_fix[i].value();
    xu_sub_tape[i] <<= x_fix[i].value();

  for (size_t i = 0; i < dim_u; ++i)
    // xu[dim_x + i] <<= u[i].value();
    xu_sub_tape[dim_x + i] <<= u[i].value();

  // IMPORTANT: Dont reuse x_fix here. The location of the x_fix's adoubles
  // could change and the old locations are already stored on the tape. This
  // would cause errors
  adouble_func(xu_sub_tape, xu_sub_tape + dim_x, x_fix_new, dim_x, dim_u);

  double dummy_out;
  for (size_t i = 0; i < dim_x; ++i)
    x_fix_new[i] >>= dummy_out;

  trace_off(sub_tape_num);
  // read out the result
  for (size_t i = 0; i < dim_x + dim_u; ++i)
    xu[i] = xu_sub_tape[i];

  delete[] xu;
  delete[] xu_sub_tape;
  delete[] x_fix_new;

  // reset default tape
  setDefaultTapeId(last_default_tape_id);
  return k;
}
