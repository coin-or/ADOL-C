#ifndef ADOLC_TL_INTERFACE_H
#define ADOLC_TL_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif
// Used to handle Tape-Less adouble as void*
typedef void *TLAdoubleHandle;
#ifdef __cplusplus
}
#endif

/*
Constructor & Destructor for class tape-less adouble
*/
#ifdef __cplusplus
extern "C" {
#endif
// Constructor
TLAdoubleHandle create_tl_adouble(const double x);
TLAdoubleHandle create_tl_adouble_with_ad(const double val,
                                          const double *ad_val);
TLAdoubleHandle create_tl_adouble_empty();

// Destructor
void free_tl_adouble(TLAdoubleHandle a);
#ifdef __cplusplus
}
#endif

/*
Utilities for tape-less adouble
*/
#ifdef __cplusplus
extern "C" {
#endif
void set_num_dir(const size_t n);
void set_tl_value(TLAdoubleHandle a, const double val);
void set_tl_ad_value(TLAdoubleHandle a, const double *const val);
void set_tl_ad_value_idx(TLAdoubleHandle a, const size_t pos, const double val);
double get_tl_value(TLAdoubleHandle a);
const double *get_tl_ad_values(TLAdoubleHandle a);
double get_tl_ad_value_idx(TLAdoubleHandle a, const size_t pos);
#ifdef __cplusplus
}
#endif

/*
Arithmetics for class tape-less adouble
*/
#ifdef __cplusplus
extern "C" {
#endif
TLAdoubleHandle add_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
TLAdoubleHandle add_double_tl_adouble(const double x, TLAdoubleHandle b);
TLAdoubleHandle add_tl_adouble_double(TLAdoubleHandle a, const double x);

TLAdoubleHandle mult_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
TLAdoubleHandle mult_double_tl_adouble(const double x, TLAdoubleHandle b);
TLAdoubleHandle mult_tl_adouble_double(TLAdoubleHandle a, const double x);

TLAdoubleHandle subtr_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
TLAdoubleHandle subtr_double_tl_adouble(const double x, TLAdoubleHandle b);
TLAdoubleHandle subtr_tl_adouble_double(TLAdoubleHandle a, const double x);

TLAdoubleHandle div_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
TLAdoubleHandle div_double_tl_adouble(const double x, TLAdoubleHandle b);
TLAdoubleHandle div_tl_adouble_double(TLAdoubleHandle a, const double x);

TLAdoubleHandle max_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
TLAdoubleHandle max_double_tl_adouble(const double x, TLAdoubleHandle b);
TLAdoubleHandle max_tl_adouble_double(TLAdoubleHandle a, const double x);

TLAdoubleHandle min_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
TLAdoubleHandle min_double_tl_adouble(const double x, TLAdoubleHandle b);
TLAdoubleHandle min_tl_adouble_double(TLAdoubleHandle a, const double x);

TLAdoubleHandle pow_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
TLAdoubleHandle pow_tl_adouble_double(TLAdoubleHandle a, const double x);

bool ge_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
bool ge_double_tl_adouble(const double x, TLAdoubleHandle b);
bool ge_tl_adouble_double(TLAdoubleHandle a, const double x);

bool g_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
bool g_double_tl_adouble(const double x, TLAdoubleHandle b);
bool g_tl_adouble_double(TLAdoubleHandle a, const double x);

bool le_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
bool le_double_tl_adouble(const double x, TLAdoubleHandle b);
bool le_tl_adouble_double(TLAdoubleHandle a, const double x);

bool l_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
bool l_double_tl_adouble(const double x, TLAdoubleHandle b);
bool l_tl_adouble_double(TLAdoubleHandle a, const double x);

bool eq_tl_adouble(TLAdoubleHandle a, TLAdoubleHandle b);
bool eq_double_tl_adouble(const double x, TLAdoubleHandle b);
bool eq_tl_adouble_double(TLAdoubleHandle a, const double x);

TLAdoubleHandle tl_abs(TLAdoubleHandle a);
TLAdoubleHandle tl_sqrt(TLAdoubleHandle a);
TLAdoubleHandle tl_log(TLAdoubleHandle a);
TLAdoubleHandle tl_log10(TLAdoubleHandle a);
TLAdoubleHandle tl_sin(TLAdoubleHandle a);
TLAdoubleHandle tl_cos(TLAdoubleHandle a);
TLAdoubleHandle tl_tan(TLAdoubleHandle a);
TLAdoubleHandle tl_exp(TLAdoubleHandle a);
TLAdoubleHandle tl_asin(TLAdoubleHandle a);
TLAdoubleHandle tl_acos(TLAdoubleHandle a);
TLAdoubleHandle tl_atan(TLAdoubleHandle a);
TLAdoubleHandle tl_sinh(TLAdoubleHandle a);
TLAdoubleHandle tl_cosh(TLAdoubleHandle a);
TLAdoubleHandle tl_tanh(TLAdoubleHandle a);
TLAdoubleHandle tl_asinh(TLAdoubleHandle a);
TLAdoubleHandle tl_acosh(TLAdoubleHandle a);
TLAdoubleHandle tl_atanh(TLAdoubleHandle a);
TLAdoubleHandle tl_ceil(TLAdoubleHandle a);
TLAdoubleHandle tl_floor(TLAdoubleHandle a);
TLAdoubleHandle tl_ldexp(TLAdoubleHandle a, const int n);
TLAdoubleHandle tl_erf(TLAdoubleHandle a);
TLAdoubleHandle tl_erfc(TLAdoubleHandle a);
#ifdef __cplusplus
}
#endif

#endif // ADOLC_TL_INTERFACE_H
