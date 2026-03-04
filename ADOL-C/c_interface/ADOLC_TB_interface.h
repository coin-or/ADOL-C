#ifndef ADOLC_TB_INTERFACE_H
#define ADOLC_TB_INTERFACE_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t loc;
} tape_loc;

#define ADOLC_TB_INVALID_LOC ((size_t)-1)

tape_loc adolc_tb_new(double x);
tape_loc adolc_tb_new_empty(void);
void adolc_tb_free(tape_loc v);
void adolc_tb_free_on_tape(short tape_id, tape_loc v);
double adolc_tb_value(tape_loc v);
void adolc_tb_set_value(tape_loc v, double x);

tape_loc adolc_tb_assign(double x);
void adolc_tb_assign_to(tape_loc out, tape_loc in);
void adolc_tb_assign_double_to(tape_loc out, double x);

tape_loc adolc_tb_add(tape_loc a, tape_loc b);
tape_loc adolc_tb_sub(tape_loc a, tape_loc b);
tape_loc adolc_tb_mul(tape_loc a, tape_loc b);
tape_loc adolc_tb_div(tape_loc a, tape_loc b);
tape_loc adolc_tb_pow(tape_loc a, tape_loc b);
tape_loc adolc_tb_min(tape_loc a, tape_loc b);
tape_loc adolc_tb_max(tape_loc a, tape_loc b);

tape_loc adolc_tb_add_d(tape_loc a, double x);
tape_loc adolc_tb_sub_d(tape_loc a, double x);
tape_loc adolc_tb_mul_d(tape_loc a, double x);
tape_loc adolc_tb_div_d(tape_loc a, double x);
tape_loc adolc_tb_pow_d(tape_loc a, double x);
tape_loc adolc_tb_min_d(tape_loc a, double x);
tape_loc adolc_tb_max_d(tape_loc a, double x);
tape_loc adolc_tb_d_add(double x, tape_loc a);
tape_loc adolc_tb_d_sub(double x, tape_loc a);
tape_loc adolc_tb_d_mul(double x, tape_loc a);
tape_loc adolc_tb_d_div(double x, tape_loc a);
tape_loc adolc_tb_d_min(double x, tape_loc a);
tape_loc adolc_tb_d_max(double x, tape_loc a);

tape_loc adolc_tb_abs(tape_loc a);
tape_loc adolc_tb_sqrt(tape_loc a);
tape_loc adolc_tb_log(tape_loc a);
tape_loc adolc_tb_log10(tape_loc a);
tape_loc adolc_tb_sin(tape_loc a);
tape_loc adolc_tb_cos(tape_loc a);
tape_loc adolc_tb_tan(tape_loc a);
tape_loc adolc_tb_exp(tape_loc a);
tape_loc adolc_tb_asin(tape_loc a);
tape_loc adolc_tb_acos(tape_loc a);
tape_loc adolc_tb_atan(tape_loc a);
tape_loc adolc_tb_sinh(tape_loc a);
tape_loc adolc_tb_cosh(tape_loc a);
tape_loc adolc_tb_tanh(tape_loc a);
tape_loc adolc_tb_asinh(tape_loc a);
tape_loc adolc_tb_acosh(tape_loc a);
tape_loc adolc_tb_atanh(tape_loc a);
tape_loc adolc_tb_ceil(tape_loc a);
tape_loc adolc_tb_floor(tape_loc a);
tape_loc adolc_tb_ldexp(tape_loc a, int n);
tape_loc adolc_tb_erf(tape_loc a);
tape_loc adolc_tb_erfc(tape_loc a);

void adolc_tb_add_to(tape_loc out, tape_loc rhs);
void adolc_tb_sub_to(tape_loc out, tape_loc rhs);
void adolc_tb_mul_to(tape_loc out, tape_loc rhs);
void adolc_tb_div_to(tape_loc out, tape_loc rhs);

bool adolc_tb_ge(tape_loc a, tape_loc b);
bool adolc_tb_g(tape_loc a, tape_loc b);
bool adolc_tb_le(tape_loc a, tape_loc b);
bool adolc_tb_l(tape_loc a, tape_loc b);
bool adolc_tb_eq(tape_loc a, tape_loc b);
bool adolc_tb_ge_d(tape_loc a, double x);
bool adolc_tb_g_d(tape_loc a, double x);
bool adolc_tb_le_d(tape_loc a, double x);
bool adolc_tb_l_d(tape_loc a, double x);
bool adolc_tb_eq_d(tape_loc a, double x);
bool adolc_tb_d_ge(double x, tape_loc a);
bool adolc_tb_d_g(double x, tape_loc a);
bool adolc_tb_d_le(double x, tape_loc a);
bool adolc_tb_d_l(double x, tape_loc a);
bool adolc_tb_d_eq(double x, tape_loc a);

int adolc_trace_on(short tag, int keep);
void adolc_trace_off(int flag);
void adolc_ensure_tape(short tape_id);
void adolc_tb_independent(tape_loc v, double x);
void adolc_tb_dependent(tape_loc v, double *y);
size_t adolc_num_independent(short tape_id);
size_t adolc_num_dependent(short tape_id);
void adolc_enable_min_max_using_abs(void);
void adolc_disable_min_max_using_abs(void);
tape_loc adolc_mkparam(double val);
void adolc_set_param_vec(short tape_id, unsigned int numparam,
                         const double *paramvec);

#ifdef __cplusplus
}
#endif

#endif // ADOLC_TB_INTERFACE_H
