#ifndef ADOLC_TB_INTERFACE_H
#define ADOLC_TB_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif
// Used to handle Tape-based adouble as void*
typedef void *TBAdoubleHandle;
#ifdef __cplusplus
}
#endif

/*
Constructor & Destructor for class adouble
*/
#ifdef __cplusplus
extern "C" {
#endif
// Constructor
TBAdoubleHandle create_tb_adouble(const double x);
TBAdoubleHandle create_tb_adouble_empty();

// Destructor
void free_tb_adouble(TBAdoubleHandle a);
#ifdef __cplusplus
}
#endif

/*
Utilities for adouble
*/
#ifdef __cplusplus
extern "C" {
#endif
double get_tb_value(TBAdoubleHandle a);
#ifdef __cplusplus
}
#endif

/*
Arithmetics for class tb adouble
*/
#ifdef __cplusplus
extern "C" {
#endif
TBAdoubleHandle add_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
TBAdoubleHandle add_double_tb_adouble(const double x, TBAdoubleHandle b);
TBAdoubleHandle add_tb_adouble_double(TBAdoubleHandle a, const double x);

TBAdoubleHandle mult_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
TBAdoubleHandle mult_double_tb_adouble(const double x, TBAdoubleHandle b);
TBAdoubleHandle mult_tb_adouble_double(TBAdoubleHandle a, const double x);

TBAdoubleHandle subtr_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
TBAdoubleHandle subtr_double_tb_adouble(const double x, TBAdoubleHandle b);
TBAdoubleHandle subtr_tb_adouble_double(TBAdoubleHandle a, const double x);

TBAdoubleHandle div_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
TBAdoubleHandle div_double_tb_adouble(const double x, TBAdoubleHandle b);
TBAdoubleHandle div_tb_adouble_double(TBAdoubleHandle a, const double x);

TBAdoubleHandle max_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
TBAdoubleHandle max_double_tb_adouble(const double x, TBAdoubleHandle b);
TBAdoubleHandle max_tb_adouble_double(TBAdoubleHandle a, const double x);

TBAdoubleHandle min_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
TBAdoubleHandle min_double_tb_adouble(const double x, TBAdoubleHandle b);
TBAdoubleHandle min_tb_adouble_double(TBAdoubleHandle a, const double x);

TBAdoubleHandle pow_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
TBAdoubleHandle pow_tb_adouble_double(TBAdoubleHandle a, const double x);

bool ge_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
bool ge_double_tb_adouble(const double x, TBAdoubleHandle b);
bool ge_tb_adouble_double(TBAdoubleHandle a, const double x);

bool g_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
bool g_double_tb_adouble(const double x, TBAdoubleHandle b);
bool g_tb_adouble_double(TBAdoubleHandle a, const double x);

bool le_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
bool le_double_tb_adouble(const double x, TBAdoubleHandle b);
bool le_tb_adouble_double(TBAdoubleHandle a, const double x);

bool l_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
bool l_double_tb_adouble(const double x, TBAdoubleHandle b);
bool l_tb_adouble_double(TBAdoubleHandle a, const double x);

bool eq_tb_adouble(TBAdoubleHandle a, TBAdoubleHandle b);
bool eq_double_tb_adouble(const double x, TBAdoubleHandle b);
bool eq_tb_adouble_double(TBAdoubleHandle a, const double x);

TBAdoubleHandle tb_abs(TBAdoubleHandle a);
TBAdoubleHandle tb_sqrt(TBAdoubleHandle a);
TBAdoubleHandle tb_log(TBAdoubleHandle a);
TBAdoubleHandle tb_log10(TBAdoubleHandle a);
TBAdoubleHandle tb_sin(TBAdoubleHandle a);
TBAdoubleHandle tb_cos(TBAdoubleHandle a);
TBAdoubleHandle tb_tan(TBAdoubleHandle a);
TBAdoubleHandle tb_exp(TBAdoubleHandle a);
TBAdoubleHandle tb_asin(TBAdoubleHandle a);
TBAdoubleHandle tb_acos(TBAdoubleHandle a);
TBAdoubleHandle tb_atan(TBAdoubleHandle a);
TBAdoubleHandle tb_sinh(TBAdoubleHandle a);
TBAdoubleHandle tb_cosh(TBAdoubleHandle a);
TBAdoubleHandle tb_tanh(TBAdoubleHandle a);
TBAdoubleHandle tb_asinh(TBAdoubleHandle a);
TBAdoubleHandle tb_acosh(TBAdoubleHandle a);
TBAdoubleHandle tb_atanh(TBAdoubleHandle a);
TBAdoubleHandle tb_ceil(TBAdoubleHandle a);
TBAdoubleHandle tb_floor(TBAdoubleHandle a);
TBAdoubleHandle tb_ldexp(TBAdoubleHandle a, int n);
TBAdoubleHandle tb_erf(TBAdoubleHandle a);
TBAdoubleHandle tb_erfc(TBAdoubleHandle a);
#ifdef __cplusplus
}
#endif

/*
Tape utilities
*/
#ifdef __cplusplus
extern "C" {
#endif
int c_trace_on(short int tag, int keep = 0);
void c_trace_off(int flag = 0);
void create_independent(TBAdoubleHandle a, const double x);
void create_dependent(TBAdoubleHandle a, double *y);
size_t num_independent(short tapeId);
size_t num_dependent(short tapeId);
void enable_min_max_using_abs();
void disable_min_max_using_abs();
TBAdoubleHandle mkparam_(const double val);
#ifdef __cplusplus
}
#endif

#endif // ADOLC_TB_INTERFACE_H
