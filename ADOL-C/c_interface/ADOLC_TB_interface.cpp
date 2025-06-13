#include "ADOLC_TB_interface.h"
#include <adolc/adolc.h>

/*
Constructor & Destructor for class tape-based adouble
*/
extern "C" {
TBAdoubleHandle create_tb_adouble(const double x) { return new adouble(x); }
TBAdoubleHandle create_tb_adouble_empty() { return new adouble(); }

void free_tb_adouble(TBAdoubleHandle a) {
  return delete static_cast<adouble *>(a);
}
}

/*
Utilities for adouble
*/
extern "C" {
double get_tb_value(TBAdoubleHandle a) {
  return static_cast<adouble *>(a)->getValue();
}
}

/*
Arithmetics for class adouble
*/
extern "C" {
TBAdoubleHandle add_tb_adouble(const TBAdoubleHandle a,
                               const TBAdoubleHandle b) {
  return new adouble(*static_cast<adouble *>(a) + *static_cast<adouble *>(b));
}
TBAdoubleHandle add_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return new adouble(x + *static_cast<adouble *>(b));
}
TBAdoubleHandle add_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return new adouble(*static_cast<adouble *>(a) + x);
}
TBAdoubleHandle mult_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return new adouble(*static_cast<adouble *>(a) * *static_cast<adouble *>(b));
}
TBAdoubleHandle mult_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return new adouble(x * *static_cast<adouble *>(b));
}
TBAdoubleHandle mult_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return new adouble(*static_cast<adouble *>(a) * x);
}
TBAdoubleHandle subtr_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return new adouble(*static_cast<adouble *>(a) - *static_cast<adouble *>(b));
}
TBAdoubleHandle subtr_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return new adouble(x - *static_cast<adouble *>(b));
}
TBAdoubleHandle subtr_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return new adouble(*static_cast<adouble *>(a) - x);
}
TBAdoubleHandle div_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return new adouble(*static_cast<adouble *>(a) / *static_cast<adouble *>(b));
}
TBAdoubleHandle div_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return new adouble(x / *static_cast<adouble *>(b));
}
TBAdoubleHandle div_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return new adouble(*static_cast<adouble *>(a) / x);
}
TBAdoubleHandle max_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return new adouble(
      fmax(*static_cast<adouble *>(a), *static_cast<adouble *>(b)));
}
TBAdoubleHandle max_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return new adouble(fmax(x, *static_cast<adouble *>(b)));
}
TBAdoubleHandle max_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return new adouble(fmax(*static_cast<adouble *>(a), x));
}
TBAdoubleHandle min_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return new adouble(
      fmin(*static_cast<adouble *>(a), *static_cast<adouble *>(b)));
}
TBAdoubleHandle min_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return new adouble(fmin(x, *static_cast<adouble *>(b)));
}
TBAdoubleHandle min_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return new adouble(fmin(*static_cast<adouble *>(a), x));
}
TBAdoubleHandle pow_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return new adouble(
      pow(*static_cast<adouble *>(a), *static_cast<adouble *>(b)));
}
TBAdoubleHandle pow_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return new adouble(pow(*static_cast<adouble *>(a), x));
}
bool ge_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return *static_cast<adouble *>(a) >= *static_cast<adouble *>(b);
}
bool ge_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return x >= *static_cast<adouble *>(b);
}
bool ge_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return *static_cast<adouble *>(a) >= x;
}
bool g_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return *static_cast<adouble *>(a) > *static_cast<adouble *>(b);
}
bool g_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return x > *static_cast<adouble *>(b);
}
bool g_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return *static_cast<adouble *>(a) > x;
}
bool le_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return *static_cast<adouble *>(a) <= *static_cast<adouble *>(b);
}
bool le_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return x <= *static_cast<adouble *>(b);
}
bool le_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return *static_cast<adouble *>(a) <= x;
}
bool l_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return *static_cast<adouble *>(a) < *static_cast<adouble *>(b);
}
bool l_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return x < *static_cast<adouble *>(b);
}
bool l_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return *static_cast<adouble *>(a) < x;
}
bool eq_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b) {
  return *static_cast<adouble *>(a) == *static_cast<adouble *>(b);
}
bool eq_double_tb_adouble(const double x, TBAdoubleHandle b) {
  return x == *static_cast<adouble *>(b);
}
bool eq_tb_adouble_double(TBAdoubleHandle a, const double x) {
  return *static_cast<adouble *>(a) == x;
}
TBAdoubleHandle tb_abs(TBAdoubleHandle a) {
  return new adouble(fabs(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_sqrt(TBAdoubleHandle a) {
  return new adouble(sqrt(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_log(TBAdoubleHandle a) {
  return new adouble(log(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_log10(TBAdoubleHandle a) {
  return new adouble(log10(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_sin(TBAdoubleHandle a) {
  return new adouble(sin(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_cos(TBAdoubleHandle a) {
  return new adouble(cos(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_tan(TBAdoubleHandle a) {
  return new adouble(tan(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_exp(TBAdoubleHandle a) {
  return new adouble(exp(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_asin(TBAdoubleHandle a) {
  return new adouble(asin(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_acos(TBAdoubleHandle a) {
  return new adouble(acos(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_atan(TBAdoubleHandle a) {
  return new adouble(atan(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_sinh(TBAdoubleHandle a) {
  return new adouble(sinh(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_cosh(TBAdoubleHandle a) {
  return new adouble(cosh(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_tanh(TBAdoubleHandle a) {
  return new adouble(tanh(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_asinh(TBAdoubleHandle a) {
  return new adouble(asinh(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_acosh(TBAdoubleHandle a) {
  return new adouble(acosh(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_atanh(TBAdoubleHandle a) {
  return new adouble(atanh(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_ceil(TBAdoubleHandle a) {
  return new adouble(ceil(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_floor(TBAdoubleHandle a) {
  return new adouble(floor(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_ldexp(TBAdoubleHandle a, const int n) {
  return new adouble(ldexp(*static_cast<adouble *>(a), n));
}
TBAdoubleHandle tb_erf(TBAdoubleHandle a) {
  return new adouble(erf(*static_cast<adouble *>(a)));
}
TBAdoubleHandle tb_erfc(TBAdoubleHandle a) {
  return new adouble(erfc(*static_cast<adouble *>(a)));
}
}

/*
Tape utilities
*/
extern "C" {
int c_trace_on(short int tag, int keep) { return trace_on(tag, keep); }
void c_trace_off(int flag) { return trace_off(flag); }
void create_independent(TBAdoubleHandle a, const double x) {
  *static_cast<adouble *>(a) <<= x;
}
void create_dependent(TBAdoubleHandle a, double *y) {
  *static_cast<adouble *>(a) >>= *y;
}
size_t num_independent(short tapeId) {
  size_t y[TapeInfos::STAT_SIZE];
  tapestats(tapeId, y);
  return y[TapeInfos::NUM_INDEPENDENTS];
}
size_t num_dependent(short tapeId) {
  size_t y[TapeInfos::STAT_SIZE];
  tapestats(tapeId, y);
  return y[TapeInfos::NUM_DEPENDENTS];
}
void enable_min_max_using_abs() { return enableMinMaxUsingAbs(); }
void disable_min_max_using_abs() { return disableMinMaxUsingAbs(); }
TBAdoubleHandle mkparam_(const double val) {
  return new adouble(pdouble::mkparam(val));
}
}
