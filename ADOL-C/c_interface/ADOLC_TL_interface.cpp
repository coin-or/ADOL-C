#include "ADOLC_TL_interface.h"
#include <adolc/adtl.h>

/*
Constructor & Destructor for class tape-less adouble
*/
extern "C" {
TLAdoubleHandle create_tl_adouble(const double x) {
  return new adtl::adouble(x);
}
TLAdoubleHandle create_tl_adouble_empty() { return new adtl::adouble(); }
TLAdoubleHandle create_tl_adouble_with_ad(const double val,
                                          const double *ad_val) {
  return new adtl::adouble(val, ad_val);
}
void free_tl_adouble(TLAdoubleHandle a) {
  return delete static_cast<adtl::adouble *>(a);
}
}

/*
Utilities for adouble
*/
extern "C" {
void set_num_dir(const size_t n) { return adtl::setNumDir(n); }

void set_tl_value(TLAdoubleHandle a, const double val) {
  return static_cast<adtl::adouble *>(a)->setValue(val);
}

void set_tl_ad_value(TLAdoubleHandle a, const double *const val) {
  return static_cast<adtl::adouble *>(a)->setADValue(val);
}

void set_tl_ad_value_idx(TLAdoubleHandle a, const size_t pos,
                         const double val) {
  return static_cast<adtl::adouble *>(a)->setADValue(pos, val);
}

double get_tl_value(const TLAdoubleHandle a) {
  return static_cast<adtl::adouble *>(a)->getValue();
}
const double *get_tl_ad_values(const TLAdoubleHandle a) {
  return (static_cast<adtl::adouble *>(a))->getADValue();
}
double get_tl_ad_value_idx(const TLAdoubleHandle a, const size_t pos) {
  return static_cast<adtl::adouble *>(a)->getADValue(pos);
}
}

/*
Arithmetics for class adouble
*/
extern "C" {
TLAdoubleHandle add_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return new adtl::adouble(*static_cast<adtl::adouble *>(a) +
                           *static_cast<adtl::adouble *>(b));
}
TLAdoubleHandle add_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return new adtl::adouble(x + *static_cast<adtl::adouble *>(b));
}
TLAdoubleHandle add_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return new adtl::adouble(*static_cast<adtl::adouble *>(a) + x);
}
TLAdoubleHandle mult_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return new adtl::adouble(*static_cast<adtl::adouble *>(a) *
                           *static_cast<adtl::adouble *>(b));
}
TLAdoubleHandle mult_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return new adtl::adouble(x * *static_cast<adtl::adouble *>(b));
}
TLAdoubleHandle mult_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return new adtl::adouble(*static_cast<adtl::adouble *>(a) * x);
}
TLAdoubleHandle subtr_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return new adtl::adouble(*static_cast<adtl::adouble *>(a) -
                           *static_cast<adtl::adouble *>(b));
}
TLAdoubleHandle subtr_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return new adtl::adouble(x - *static_cast<adtl::adouble *>(b));
}
TLAdoubleHandle subtr_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return new adtl::adouble(*static_cast<adtl::adouble *>(a) - x);
}
TLAdoubleHandle div_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return new adtl::adouble(*static_cast<adtl::adouble *>(a) /
                           *static_cast<adtl::adouble *>(b));
}
TLAdoubleHandle div_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return new adtl::adouble(x / *static_cast<adtl::adouble *>(b));
}
TLAdoubleHandle div_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return new adtl::adouble(*static_cast<adtl::adouble *>(a) / x);
}
TLAdoubleHandle max_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return new adtl::adouble(
      fmax(*static_cast<adtl::adouble *>(a), *static_cast<adtl::adouble *>(b)));
}
TLAdoubleHandle max_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return new adtl::adouble(fmax(x, *static_cast<adtl::adouble *>(b)));
}
TLAdoubleHandle max_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return new adtl::adouble(fmax(*static_cast<adtl::adouble *>(a), x));
}
TLAdoubleHandle min_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return new adtl::adouble(
      fmin(*static_cast<adtl::adouble *>(a), *static_cast<adtl::adouble *>(b)));
}
TLAdoubleHandle min_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return new adtl::adouble(fmin(x, *static_cast<adtl::adouble *>(b)));
}
TLAdoubleHandle min_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return new adtl::adouble(fmin(*static_cast<adtl::adouble *>(a), x));
}
TLAdoubleHandle pow_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return new adtl::adouble(
      pow(*static_cast<adtl::adouble *>(a), *static_cast<adtl::adouble *>(b)));
}
TLAdoubleHandle pow_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return new adtl::adouble(pow(*static_cast<adtl::adouble *>(a), x));
}
bool ge_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return *static_cast<adtl::adouble *>(a) >= *static_cast<adtl::adouble *>(b);
}
bool ge_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return x >= *static_cast<adtl::adouble *>(b);
}
bool ge_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return *static_cast<adtl::adouble *>(a) >= x;
}
bool g_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return *static_cast<adtl::adouble *>(a) > *static_cast<adtl::adouble *>(b);
}
bool g_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return x > *static_cast<adtl::adouble *>(b);
}
bool g_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return *static_cast<adtl::adouble *>(a) > x;
}
bool le_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return *static_cast<adtl::adouble *>(a) <= *static_cast<adtl::adouble *>(b);
}
bool le_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return x <= *static_cast<adtl::adouble *>(b);
}
bool le_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return *static_cast<adtl::adouble *>(a) <= x;
}
bool l_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return *static_cast<adtl::adouble *>(a) < *static_cast<adtl::adouble *>(b);
}
bool l_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return x < *static_cast<adtl::adouble *>(b);
}
bool l_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return *static_cast<adtl::adouble *>(a) < x;
}
bool eq_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b) {
  return *static_cast<adtl::adouble *>(a) == *static_cast<adtl::adouble *>(b);
}
bool eq_double_tl_adouble(const double x, TLAdoubleHandle b) {
  return x == *static_cast<adtl::adouble *>(b);
}
bool eq_tl_adouble_double(TLAdoubleHandle a, const double x) {
  return *static_cast<adtl::adouble *>(a) == x;
}
TLAdoubleHandle tl_abs(TLAdoubleHandle a) {
  return new adtl::adouble(fabs(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_sqrt(TLAdoubleHandle a) {
  return new adtl::adouble(sqrt(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_log(TLAdoubleHandle a) {
  return new adtl::adouble(log(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_log10(TLAdoubleHandle a) {
  return new adtl::adouble(log10(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_sin(TLAdoubleHandle a) {
  return new adtl::adouble(sin(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_cos(TLAdoubleHandle a) {
  return new adtl::adouble(cos(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_tan(TLAdoubleHandle a) {
  return new adtl::adouble(tan(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_exp(TLAdoubleHandle a) {
  return new adtl::adouble(exp(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_asin(TLAdoubleHandle a) {
  return new adtl::adouble(asin(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_acos(TLAdoubleHandle a) {
  return new adtl::adouble(acos(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_atan(TLAdoubleHandle a) {
  return new adtl::adouble(atan(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_sinh(TLAdoubleHandle a) {
  return new adtl::adouble(sinh(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_cosh(TLAdoubleHandle a) {
  return new adtl::adouble(cosh(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_tanh(TLAdoubleHandle a) {
  return new adtl::adouble(tanh(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_asinh(TLAdoubleHandle a) {
  return new adtl::adouble(asinh(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_acosh(TLAdoubleHandle a) {
  return new adtl::adouble(acosh(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_atanh(TLAdoubleHandle a) {
  return new adtl::adouble(atanh(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_ceil(TLAdoubleHandle a) {
  return new adtl::adouble(ceil(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_floor(TLAdoubleHandle a) {
  return new adtl::adouble(floor(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_ldexp(TLAdoubleHandle a, const int n) {
  return new adtl::adouble(ldexp(*static_cast<adtl::adouble *>(a), n));
}
TLAdoubleHandle tl_erf(TLAdoubleHandle a) {
  return new adtl::adouble(erf(*static_cast<adtl::adouble *>(a)));
}
TLAdoubleHandle tl_erfc(TLAdoubleHandle a) {
  return new adtl::adouble(erfc(*static_cast<adtl::adouble *>(a)));
}
}
