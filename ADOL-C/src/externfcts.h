/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts.h
 Revision: $Id$
 Contents: public functions and data types for extern (differentiated)
           functions.
 
 Copyright (c) Andreas Kowarz

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
                     
----------------------------------------------------------------------------*/

#if !defined(ADOLC_EXTERNFCTS_H)
#define ADOLC_EXTERNFCTS_H 1

#include <adolc/common.h>
#include <adolc/adouble.h>

BEGIN_C_DECLS

typedef int (ADOLC_ext_fct) (int n, double *x, int m, double *y);
typedef int (ADOLC_ext_fct_fos_forward) (int n, double *dp_x, double *dp_X, int m, double *dp_y, double *dp_Y);
typedef int (ADOLC_ext_fct_fov_forward) (int n, double *dp_x, int p, double **dpp_X, int m, double *dp_y, double **dpp_Y);
typedef int (ADOLC_ext_fct_hos_forward) (int n, double *dp_x, int d, double **dpp_X, int m, double *dp_y, double **dpp_Y);
typedef int (ADOLC_ext_fct_hov_forward) (int n, double *dp_x, int d, int p, double ***dppp_X, int m, double *dp_y, double ***dppp_Y);
typedef int (ADOLC_ext_fct_fos_reverse) (int m, double *dp_U, int n, double *dp_Z, double *dp_x, double *dp_y);
typedef int (ADOLC_ext_fct_fov_reverse) (int m, int p, double **dpp_U, int n, double **dpp_Z);
typedef int (ADOLC_ext_fct_hos_reverse) (int m, double *dp_U, int n, int d, double **dpp_Z); 
typedef int (ADOLC_ext_fct_hov_reverse) (int m, int p, double **dpp_U, int n, int d, double ***dppp_Z, short **spp_nz);


/**
 * A variable of this type has to be instantiated by reg_ext_fct (see below) and a pointer to it is
 * returned. Within reg_ext_fct the memberse function and index are properly set. 
 * is likely to be wrong in this case. Use pointers instead. 
 */
typedef struct {

  /**
   * DO NOT touch - the function pointer is set through reg_ext_fct
   */
  ADOLC_ext_fct *function;  

  /**
   * DO NOT touch - the index is set through reg_ext_fct
   */
  locint index;            

  /** 
   * below are function pointers used for call back from the corresponding ADOL-C trace interpreters; 
   * these function pointers are initialized to 0 by reg_ext_fct; 
   * the  user needs to set eplicitly the function pointers for the trace interpreters called in the 
   * application driver
   */

  /**
   * this points to a  method implementing a forward execution of the externally differentiated function dp_y=f(dp_x); 
   * the pointer would typically be set to the same function pointer supplied in the call to reg_ext_fct, 
   * i.e. zos_forward would be equal to function (above) 
   * but there are cases when it makes sense for this to be different as illustrated
   * in examples/additional_examples/ext_diff_func/ext_diff_func.cpp  
   */
  ADOLC_ext_fct *zos_forward;

  /**
   * this points to a  method implementing a forward execution of the externally differentiated function dp_y=f(dp_x)
   * and computing the projection dp_Y=Jacobian*dp_x 
   * see also the explanation of the dp_X/Y  members below.
   */
  ADOLC_ext_fct_fos_forward *fos_forward;

  /**
   * this points to a  method implementing a forward execution of the externally differentiated function dp_y=f(dp_x)
   * and computing the projection dpp_Y=Jacobian*dpp_x 
   * see also the explanation of the dpp_X/Y  members below.
   */
  ADOLC_ext_fct_fov_forward *fov_forward;
  /** 
   * higher order scalar forward for external functions  is currently not implemented in uni5_for.c
   */
  ADOLC_ext_fct_hos_forward *hos_forward; 
  /** 
   * higher order vector forward for external functions  is currently not implemented in uni5_for.c
   */
  ADOLC_ext_fct_hov_forward *hov_forward;
  /**
   * this points to a  method computing the projection dp_Z=transpose(dp_U) * Jacobian
   * see also the explanation of the dp_U/Z  members below.
   */
  ADOLC_ext_fct_fos_reverse *fos_reverse; 
  /**
   * this points to a  method computing the projection dpp_Z=transpose(dpp_U) * Jacobian
   * see also the explanation of the dpp_U/Z  members below.
   */
  ADOLC_ext_fct_fov_reverse *fov_reverse; 
  /** 
   * higher order scalar reverse for external functions  is currently not implemented in ho_rev.c
   */
  ADOLC_ext_fct_hos_reverse *hos_reverse; 
  /** 
   * higher order vector reverse for external functions  is currently not implemented in ho_rev.c
   */
  ADOLC_ext_fct_hov_reverse *hov_reverse; 


  /**
   * The names of the variables below correspond to the formal parameters names in the call back 
   * functions above; 
   * The user has to preallocate the variables and set the pointers for any of the call back functions 
   * that will be called during trace interpretation.
   * The dimensions given below correspond to the formal arguments in the call back funtions signatures above. 
   * If the dimensions n and m change between multiple calls to the same external function, then the variables 
   * have to be preallocation with the maximum of the respective dimension values. 
   * The dp_x and dp_y pointers have to be valid during both, the tracing phase and the trace interpretation; 
   * all the other pointers are required to be valid only for the trace interpretation.
   */
       
  /** 
   * function and all _forward calls: function argument, dimension [n]
   */ 
  double *dp_x;     

  /** 
   * fos_forward: tangent direction, dimension [n]
   */ 
  double *dp_X;   

  /**
   * fov_forward: seed matrix for p directions, dimensions [n][p]
   * hos_forward: argument Taylor polynomial coefficients up to order d. dimensions [n][d] 
   */
  double **dpp_X;
  
  /**
   * hov_forward: argument Taylor polynomial coefficients up to order d in p directions. dimensions [n][p][d]
   */
  double ***dppp_X; 

  /**
   * function and all _forward calls: function result, dimension [m]
   */
  double *dp_y;   

  /**
   * fos_forward: Jacobian projection, dimension [m]
   */
  double *dp_Y;  

  /**
   * fov_forward: Jacobian projection in p directions, dimension [m][p]
   * hos_forward: result Taylor polynomial coefficients up to order d. dimensions [m][d] 
   */
  double **dpp_Y;     

  /**
   * hov_forward: result Taylor polynomial coefficients up to order d in p directions. dimensions [m][p][d]
   */
  double ***dppp_Y;

  /**
   * fos_reverse and hos_reverse:  weight vector, dimension [m]
   */
  double *dp_U;
 
  /**
   * fov_reverse and hov_reverse: p weight vectors, dimensions [p][m]
   */
  double **dpp_U;       

  /** 
   * fos_reverse: Jacobian projection, dimension [n]
   */
  double *dp_Z; 

  /** 
   * fov_reverse: Jacobian projection for p weight vectors, dimensions [p][n]
   * hos_reverse: adjoint Taylor polynomial coefficients up to order d, dimensions [n][d+1] 
   */
  double **dpp_Z;   

  /**
   * hov_reverse:  adjoint Taylor polynomial coefficients up to order d for p weight vectors, dimension [p][n][d+1]
   */
  double ***dppp_Z; 

  /** 
   * hov_reverse: non-zero pattern of dppp_Z, dimension [p][n], see also the hov_reverse ADOL-C driver 
   */
  short **spp_nz;

  /**
   * track maximal value of n when function is invoked
   */
  int max_n;

  /**
   * track maximal value of m when function is invoked
   */
  int max_m;

  /**
   * make the call such that Adol-C may be used inside
   * of the externally differentiated function;
   * defaults to non-0;
   * this implies certain storage duplication that can
   * be avoided if no nested use of Adol-C takes place
   */
  char nestedAdolc;

  /**
   * if 0, then the 'function' does not change dp_x;
   * defaults to non-0 which implies dp_x values are saved in taylors
   */
  char dp_x_changes;

  /**
   * if 0, then the value of dp_y prior to calling 'function'
   * is not required for reverse;
   * defaults to non-0 which implies  dp_y values are saved in taylors
   */
  char dp_y_priorRequired;

}
ext_diff_fct;

END_C_DECLS

#if defined(__cplusplus)
/****************************************************************************/
/*                                                          This is all C++ */

ADOLC_DLL_EXPORT ext_diff_fct *reg_ext_fct(ADOLC_ext_fct ext_fct);

ADOLC_DLL_EXPORT int call_ext_fct (ext_diff_fct *edfct,
                                   int n, double *xp, adouble *xa,
                                   int m, double *yp, adouble *ya);

#endif /* __CPLUSPLUS */

/****************************************************************************/
#endif /* ADOLC_EXTERNFCTS_H */

