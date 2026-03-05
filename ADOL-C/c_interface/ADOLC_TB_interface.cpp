#include "ADOLC_TB_interface.h"

#include <adolc/adolc.h>
#include <adolc/tape_interface.h>
#include <cassert>

namespace {

// Build a non-owning adouble wrapper around an existing tape location token.
inline adouble borrow(const tape_loc v) {
  return adouble::borrow_location(v.loc);
}

// Transfer the produced adouble location into plain C token and invalidate
// source.
inline tape_loc release_to_tape_loc(adouble &&a) { return {a.release_loc()}; }

} // namespace

extern "C" {

tape_loc adolc_tb_new(const double x) {
  adouble a(x);
  return release_to_tape_loc(std::move(a));
}

tape_loc adolc_tb_new_empty(void) {
  adouble a;
  return release_to_tape_loc(std::move(a));
}

void adolc_tb_free(const tape_loc v) {
  if (v.loc == ADOLC_TB_INVALID_LOC) {
    return;
  }
  tape_location<adouble> owning_loc(v.loc, tape_location<adouble>::OWNING);
}

void adolc_tb_free_on_tape(const short tape_id, const tape_loc v) {
  if (v.loc == ADOLC_TB_INVALID_LOC) {
    return;
  }
  assert(findTapePtr_(tape_id) != nullptr &&
         "adolc_tb_free_on_tape: tape_id does not exist");
  setCurrentTape(tape_id);
  tape_location<adouble> owning_loc(v.loc, tape_location<adouble>::OWNING);
}

double adolc_tb_value(const tape_loc v) { return borrow(v).value(); }

void adolc_tb_set_value(const tape_loc v, const double x) {
  adouble a = borrow(v);
  a = x;
}

tape_loc adolc_tb_assign(const double x) { return adolc_tb_new(x); }

void adolc_tb_assign_to(const tape_loc out, const tape_loc in) {
  adouble out_a = borrow(out);
  adouble in_a = borrow(in);
  out_a = in_a;
}

void adolc_tb_assign_double_to(const tape_loc out, const double x) {
  adouble out_a = borrow(out);
  out_a = x;
}

tape_loc adolc_tb_add(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return release_to_tape_loc(aa + bb);
}
tape_loc adolc_tb_sub(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return release_to_tape_loc(aa - bb);
}
tape_loc adolc_tb_mul(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return release_to_tape_loc(aa * bb);
}
tape_loc adolc_tb_div(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return release_to_tape_loc(aa / bb);
}
tape_loc adolc_tb_pow(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return release_to_tape_loc(pow(aa, bb));
}
tape_loc adolc_tb_min(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return release_to_tape_loc(fmin(aa, bb));
}
tape_loc adolc_tb_max(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return release_to_tape_loc(fmax(aa, bb));
}

tape_loc adolc_tb_add_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return release_to_tape_loc(aa + x);
}
tape_loc adolc_tb_sub_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return release_to_tape_loc(aa - x);
}
tape_loc adolc_tb_mul_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return release_to_tape_loc(aa * x);
}
tape_loc adolc_tb_div_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return release_to_tape_loc(aa / x);
}
tape_loc adolc_tb_pow_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return release_to_tape_loc(pow(aa, x));
}
tape_loc adolc_tb_min_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return release_to_tape_loc(fmin(aa, x));
}
tape_loc adolc_tb_max_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return release_to_tape_loc(fmax(aa, x));
}
tape_loc adolc_tb_d_add(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(x + aa);
}
tape_loc adolc_tb_d_sub(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(x - aa);
}
tape_loc adolc_tb_d_mul(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(x * aa);
}
tape_loc adolc_tb_d_div(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(x / aa);
}
tape_loc adolc_tb_d_min(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(fmin(x, aa));
}
tape_loc adolc_tb_d_max(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(fmax(x, aa));
}

tape_loc adolc_tb_abs(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(fabs(aa));
}
tape_loc adolc_tb_sqrt(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(sqrt(aa));
}
tape_loc adolc_tb_log(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(log(aa));
}
tape_loc adolc_tb_log10(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(log10(aa));
}
tape_loc adolc_tb_sin(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(sin(aa));
}
tape_loc adolc_tb_cos(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(cos(aa));
}
tape_loc adolc_tb_tan(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(tan(aa));
}
tape_loc adolc_tb_exp(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(exp(aa));
}
tape_loc adolc_tb_asin(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(asin(aa));
}
tape_loc adolc_tb_acos(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(acos(aa));
}
tape_loc adolc_tb_atan(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(atan(aa));
}
tape_loc adolc_tb_sinh(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(sinh(aa));
}
tape_loc adolc_tb_cosh(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(cosh(aa));
}
tape_loc adolc_tb_tanh(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(tanh(aa));
}
tape_loc adolc_tb_asinh(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(asinh(aa));
}
tape_loc adolc_tb_acosh(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(acosh(aa));
}
tape_loc adolc_tb_atanh(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(atanh(aa));
}
tape_loc adolc_tb_ceil(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(ceil(aa));
}
tape_loc adolc_tb_floor(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(floor(aa));
}
tape_loc adolc_tb_ldexp(const tape_loc a, const int n) {
  adouble aa = borrow(a);
  return release_to_tape_loc(ldexp(aa, n));
}
tape_loc adolc_tb_erf(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(erf(aa));
}
tape_loc adolc_tb_erfc(const tape_loc a) {
  adouble aa = borrow(a);
  return release_to_tape_loc(erfc(aa));
}

void adolc_tb_add_to(const tape_loc out, const tape_loc rhs) {
  adouble out_a = borrow(out);
  adouble rhs_a = borrow(rhs);
  out_a += rhs_a;
}
void adolc_tb_sub_to(const tape_loc out, const tape_loc rhs) {
  adouble out_a = borrow(out);
  adouble rhs_a = borrow(rhs);
  out_a -= rhs_a;
}
void adolc_tb_mul_to(const tape_loc out, const tape_loc rhs) {
  adouble out_a = borrow(out);
  adouble rhs_a = borrow(rhs);
  out_a *= rhs_a;
}
void adolc_tb_div_to(const tape_loc out, const tape_loc rhs) {
  adouble out_a = borrow(out);
  adouble rhs_a = borrow(rhs);
  out_a /= rhs_a;
}

bool adolc_tb_ge(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return aa >= bb;
}
bool adolc_tb_g(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return aa > bb;
}
bool adolc_tb_le(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return aa <= bb;
}
bool adolc_tb_l(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return aa < bb;
}
bool adolc_tb_eq(const tape_loc a, const tape_loc b) {
  adouble aa = borrow(a);
  adouble bb = borrow(b);
  return aa == bb;
}
bool adolc_tb_ge_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return aa >= x;
}
bool adolc_tb_g_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return aa > x;
}
bool adolc_tb_le_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return aa <= x;
}
bool adolc_tb_l_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return aa < x;
}
bool adolc_tb_eq_d(const tape_loc a, const double x) {
  adouble aa = borrow(a);
  return aa == x;
}
bool adolc_tb_d_ge(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return x >= aa;
}
bool adolc_tb_d_g(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return x > aa;
}
bool adolc_tb_d_le(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return !(x > aa);
}
bool adolc_tb_d_l(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return x < aa;
}
bool adolc_tb_d_eq(const double x, const tape_loc a) {
  adouble aa = borrow(a);
  return aa == x;
}

int adolc_trace_on(const short tag, const int keep) {
  return trace_on(tag, keep);
}
void adolc_trace_off(const int flag) { trace_off(flag); }
void adolc_ensure_tape(const short tape_id) {
  while (findTapePtr_(tape_id) == nullptr) {
    createNewTape();
  }
}

void adolc_tb_independent(const tape_loc v, const double x) {
  adouble a = borrow(v);
  a <<= x;
}

void adolc_tb_dependent(const tape_loc v, double *y) {
  adouble a = borrow(v);
  a >>= *y;
}

size_t adolc_num_independent(const short tape_id) {
  const auto stats = tapestats(tape_id);
  return stats[TapeInfos::NUM_INDEPENDENTS];
}

size_t adolc_num_dependent(const short tape_id) {
  const auto stats = tapestats(tape_id);
  return stats[TapeInfos::NUM_DEPENDENTS];
}

void adolc_enable_min_max_using_abs(void) {
  currentTape().enableMinMaxUsingAbs();
}
void adolc_disable_min_max_using_abs(void) {
  currentTape().disableMinMaxUsingAbs();
}

tape_loc adolc_mkparam(const double val) {
  pdouble p(val);
  adouble a = adouble(p);
  return release_to_tape_loc(std::move(a));
}

void adolc_set_param_vec(const short tape_id, const unsigned int numparam,
                         const double *paramvec) {
  findTape(tape_id).set_param_vec(tape_id, numparam, paramvec);
}

} // extern "C"
