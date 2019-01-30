/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts.h
 Revision: $Id$
 Contents: public functions and data types for extern (differentiated)
           functions.
 
 Copyright (c) Andreas Kowarz, Jean Utke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
                     
----------------------------------------------------------------------------*/

#if !defined(ADOLC_EXTERNFCTS_H)
#define ADOLC_EXTERNFCTS_H 1

#include <adolc/internal/common.h>
#include <adolc/adouble.h>

BEGIN_C_DECLS

typedef int (ADOLC_ext_fct) (int n, double *x, int m, double *y);
typedef int (ADOLC_ext_fct_fos_forward) (int n, double *dp_x, double *dp_X, int m, double *dp_y, double *dp_Y);
typedef int (ADOLC_ext_fct_fov_forward) (int n, double *dp_x, int p, double **dpp_X, int m, double *dp_y, double **dpp_Y);
typedef int (ADOLC_ext_fct_hos_forward) (int n, double *dp_x, int d, double **dpp_X, int m, double *dp_y, double **dpp_Y);
typedef int (ADOLC_ext_fct_hov_forward) (int n, double *dp_x, int d, int p, double ***dppp_X, int m, double *dp_y, double ***dppp_Y);
typedef int (ADOLC_ext_fct_fos_reverse) (int m, double *dp_U, int n, double *dp_Z, double *dp_x, double *dp_y);
typedef int (ADOLC_ext_fct_fov_reverse) (int m, int p, double **dpp_U, int n, double **dpp_Z, double *dp_x, double *dp_y);
typedef int (ADOLC_ext_fct_hos_reverse) (int m, double *dp_U, int n, int d, double **dpp_Z); 
typedef int (ADOLC_ext_fct_hov_reverse) (int m, int p, double **dpp_U, int n, int d, double ***dppp_Z, short **spp_nz);

/**
 * we add a second set of function pointers with a signature expanded by a an integer array iArr
 * and a parameter iArrLength motivated by externalizing sparse solvers where the sparsity format
 * may be triples (i,j,A[i][j]) and a number of nonzero entries nz where all these integers are to
 * be packed into iArr. Doing this will still allow the integers to be stored in the locint part
 * of the tape.
 * The alternative to doing this is the introduction of a separate stack to contain the extra data
 * but this would break the self-containment of the tape.
 */
typedef int (ADOLC_ext_fct_iArr) (int iArrLength, int *iArr, int n, double *x, int m, double *y);
typedef int (ADOLC_ext_fct_iArr_fos_forward) (int iArrLength, int *iArr, int n, double *dp_x, double *dp_X, int m, double *dp_y, double *dp_Y);
typedef int (ADOLC_ext_fct_iArr_fov_forward) (int iArrLength, int *iArr, int n, double *dp_x, int p, double **dpp_X, int m, double *dp_y, double **dpp_Y);
typedef int (ADOLC_ext_fct_iArr_hos_forward) (int iArrLength, int *iArr, int n, double *dp_x, int d, double **dpp_X, int m, double *dp_y, double **dpp_Y);
typedef int (ADOLC_ext_fct_iArr_hov_forward) (int iArrLength, int *iArr, int n, double *dp_x, int d, int p, double ***dppp_X, int m, double *dp_y, double ***dppp_Y);
typedef int (ADOLC_ext_fct_iArr_fos_reverse) (int iArrLength, int *iArr, int m, double *dp_U, int n, double *dp_Z, double *dp_x, double *dp_y);
typedef int (ADOLC_ext_fct_iArr_fov_reverse) (int iArrLength, int *iArr, int m, int p, double **dpp_U, int n, double **dpp_Z, double *dp_x, double *dp_y);
typedef int (ADOLC_ext_fct_iArr_hos_reverse) (int iArrLength, int *iArr, int m, double *dp_U, int n, int d, double **dpp_Z);
typedef int (ADOLC_ext_fct_iArr_hov_reverse) (int iArrLength, int *iArr, int m, int p, double **dpp_U, int n, int d, double ***dppp_Z, short **spp_nz);


/**
 * A variable of this type has to be instantiated by reg_ext_fct (see below) and a pointer to it is
 * returned. Within reg_ext_fct the memberse function and index are properly set. 
 * is likely to be wrong in this case. Use pointers instead. 
 */
typedef struct ext_diff_fct {

  /**
   * DO NOT touch - the function pointer is set through reg_ext_fct
   */
  ADOLC_ext_fct *function;  
  ADOLC_ext_fct_iArr *function_iArr;

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
  ADOLC_ext_fct_iArr *zos_forward_iArr;

  /**
   * this points to a  method implementing a forward execution of the externally differentiated function dp_y=f(dp_x)
   * and computing the projection dp_Y=Jacobian*dp_x 
   * see also the explanation of the dp_X/Y  members below.
   */
  ADOLC_ext_fct_fos_forward *fos_forward;
  ADOLC_ext_fct_iArr_fos_forward *fos_forward_iArr;

  /**
   * this points to a  method implementing a forward execution of the externally differentiated function dp_y=f(dp_x)
   * and computing the projection dpp_Y=Jacobian*dpp_x 
   * see also the explanation of the dpp_X/Y  members below.
   */
  ADOLC_ext_fct_fov_forward *fov_forward;
  ADOLC_ext_fct_iArr_fov_forward *fov_forward_iArr;
  /** 
   * higher order scalar forward for external functions  is currently not implemented in uni5_for.c
   */
  ADOLC_ext_fct_hos_forward *hos_forward; 
  ADOLC_ext_fct_iArr_hos_forward *hos_forward_iArr;
  /** 
   * higher order vector forward for external functions  is currently not implemented in uni5_for.c
   */
  ADOLC_ext_fct_hov_forward *hov_forward;
  ADOLC_ext_fct_iArr_hov_forward *hov_forward_iArr;
  /**
   * this points to a  method computing the projection dp_Z=transpose(dp_U) * Jacobian
   * see also the explanation of the dp_U/Z  members below.
   */
  ADOLC_ext_fct_fos_reverse *fos_reverse; 
  ADOLC_ext_fct_iArr_fos_reverse *fos_reverse_iArr;
  /**
   * this points to a  method computing the projection dpp_Z=transpose(dpp_U) * Jacobian
   * see also the explanation of the dpp_U/Z  members below.
   */
  ADOLC_ext_fct_fov_reverse *fov_reverse; 
  ADOLC_ext_fct_iArr_fov_reverse *fov_reverse_iArr;
  /** 
   * higher order scalar reverse for external functions  is currently not implemented in ho_rev.c
   */
  ADOLC_ext_fct_hos_reverse *hos_reverse; 
  ADOLC_ext_fct_iArr_hos_reverse *hos_reverse_iArr;
  /** 
   * higher order vector reverse for external functions  is currently not implemented in ho_rev.c
   */
  ADOLC_ext_fct_hov_reverse *hov_reverse; 
  ADOLC_ext_fct_iArr_hov_reverse *hov_reverse_iArr;


  /**
   * The names of the variables below correspond to the formal parameters names in the call back 
   * functions above; 
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
  locint max_n;

  /**
   * track maximal value of m when function is invoked
   */
  locint max_m;

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

  /**
   * This is an all-memory pointer for allocating and deallocating
   * all other pointers can point to memory within here.
   */
  char* allmem;

  /**
   * This is a reference to an object for the C++ object-oriented
   * implementation of the external function ** do not touch **
   */
  void* obj;

  /**
   * This flag indicates that user allocates memory and internally no 
   * memory should be allocated
   */
  char user_allocated_mem;
}
ext_diff_fct;

END_C_DECLS

#include <adolc/externfcts2.h>

#if defined(__cplusplus)
/****************************************************************************/
/*                                                          This is all C++ */

ADOLC_DLL_EXPORT ext_diff_fct *reg_ext_fct(ADOLC_ext_fct ext_fct);
ADOLC_DLL_EXPORT ext_diff_fct *reg_ext_fct(ADOLC_ext_fct_iArr ext_fct);

ADOLC_DLL_EXPORT int call_ext_fct (ext_diff_fct *edfct,
                                   int n, adouble *xa,
                                   int m, adouble *ya);
ADOLC_DLL_EXPORT int call_ext_fct (ext_diff_fct *edfct,
                                   int iArrLength, int* iArr,
                                   int n, adouble *xa,
                                   int m, adouble *ya);

/**
 * zeros out the edf pointers and sets bools to defaults
 */
ADOLC_DLL_EXPORT void edf_zero(ext_diff_fct *edfct);

#include <adolc/edfclasses.h>

#endif /* __CPLUSPLUS */

/****************************************************************************/
#endif /* ADOLC_EXTERNFCTS_H */

