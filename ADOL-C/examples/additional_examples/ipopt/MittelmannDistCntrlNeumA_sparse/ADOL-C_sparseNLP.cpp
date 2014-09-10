/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     ADOL-C_sparseNLP.cpp
 Revision: $$
 Contents: class myADOLC_sparseNPL for interfacing with Ipopt
 
 Copyright (c) Andrea Walther
   
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
 This code is based on the file  MyNLP.cpp contained in the Ipopt package
 with the authors:  Carl Laird, Andreas Waechter   
----------------------------------------------------------------------------*/

/** C++ Example NLP for interfacing a problem with IPOPT and ADOL-C.
 *  MyADOL-C_sparseNLP implements a C++ example showing how to interface 
 *  with IPOPT and ADOL-C through the TNLP interface. This class 
 *  implements a distributed control problem with homogeneous
 *  Neumann boundary conditions, as formulated by Hans Mittelmann as
 *  Examples 4-6 in "Optimization Techniques for Solving Elliptic
 *  Control Problems with Control and State Constraints. Part 2:
 *  Distributed Control" taking sparsity into account.
 *
 *  exploitation of sparsity !!
 *
 */

#include <cassert>

#include "ADOL-C_sparseNLP.hpp"

#define  alpha_ 0.01 
#define  b_0j_  1.
#define  b_1j_  1.
#define  b_i0_  1.
#define  b_i1_  1.

double *y_d_;
int  N_;
double  h_;
double  hh_;

int y_index(int i, int j) 
  {
    return j + (N_+2)*i;
  }

 int u_index(int i, int j)
  {
    return (N_+2)*(N_+2) + (j-1) + (N_)*(i-1);
  }

int pde_index(int i, int j) 
  {
    return (j-1) + N_*(i-1);
  }

double y_d_cont(double x1, double x2)
  {
    return 3. + 5.*(x1*(x1-1.)*x2*(x2-1.));
  }

adouble fint_cont(double x1, double x2, adouble y, adouble u)
  {
    adouble diff_y = y-y_d_cont(x1,x2);
    return 0.5*(diff_y*diff_y + alpha_*u*u);
  }

adouble d_cont(double x1, double x2, adouble y, adouble u)
  {
    return -exp(y) - u;
  }

double fint_cont(double x1, double x2, double y, double u)
  {
    double diff_y = y-y_d_cont(x1,x2);
    return 0.5*(diff_y*diff_y + alpha_*u*u);
  }

double d_cont(double x1, double x2, double y, double u)
  {
    return -exp(y) - u;
  }

using namespace Ipopt;

/* Constructor. */
MyADOLC_sparseNLP::MyADOLC_sparseNLP()
{}

MyADOLC_sparseNLP::~MyADOLC_sparseNLP()
{}

bool MyADOLC_sparseNLP::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                         Index& nnz_h_lag, IndexStyleEnum& index_style)
{
  N_ = 15;
  h_ = (1.0/(N_+1));
  hh_= (h_*h_);

  y_d_ = new double[(N_+2)*(N_+2)];
  for (int j=0; j<= N_+1; j++) {
    for (int i=0; i<= N_+1; i++) {
      y_d_[y_index(i,j)] = y_d_cont(h_*1.*i,h_*1.*j);
    }
  }

  n = (N_+2)*(N_+2) + N_*N_;

  m = N_*N_ + 4*N_;

  generate_tapes(n, m, nnz_jac_g, nnz_h_lag);

  // use the C style indexing (0-based)
  index_style = C_STYLE;

  return true;
}

bool MyADOLC_sparseNLP::get_bounds_info(Index n, Number* x_l, Number* x_u,
                            Index m, Number* g_l, Number* g_u)
{
  // none of the variables have bounds
  for (Index i=0; i<n; i++) {
    x_l[i] = -1e20;
    x_u[i] =  1e20;
  }

  // Set the bounds for the constraints
  for (Index i=0; i<m; i++) {
    g_l[i] = 0;
    g_u[i] = 0;
  }

  return true;
}

bool MyADOLC_sparseNLP::get_starting_point(Index n, bool init_x, Number* x,
                               bool init_z, Number* z_L, Number* z_U,
                               Index m, bool init_lambda,
                               Number* lambda)
{
  // Here, we assume we only have starting values for x, if you code
  // your own NLP, you can provide starting values for the others if
  // you wish.
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);

  // set all y's to the perfect match with y_d
  for (Index i=0; i<= N_+1; i++) {
    for (Index j=0; j<= N_+1; j++) {
      x[y_index(i,j)] = y_d_[y_index(i,j)];
      //x[y_index(i,j)] += h_*x1_grid(i) + 2*h_*x2_grid(j);
    }
  }

  // Set the initial (constant) value for the u's
  for (Index i=1; i<= N_; i++) {
    for (Index j=1; j<= N_; j++) {
      x[u_index(i,j)] = 0;
    }
  }


  return true;
}

template<class T> bool  MyADOLC_sparseNLP::eval_obj(Index n, const T *x, T& obj_value)
{
  // return the value of the objective function
  obj_value = 0.;
  for (int i=1; i<=N_; i++) {
    for (int j=1; j<= N_; j++) {
      int iy = y_index(i,j);
      int iu = u_index(i,j);
      obj_value += fint_cont(h_*1.*i, h_*1.*j, x[iy], x[iu]);
    }
  }
  obj_value *= hh_;

  return true;
}

template<class T> bool  MyADOLC_sparseNLP::eval_constraints(Index n, const T *x, Index m, T* g)
{
  T val;
  // compute the discretized PDE for each interior grid point
  for (int i=1; i<=N_; i++) {
    for (int j=1; j<=N_; j++) {

      // Start with the discretized Laplacian operator
      val = 4.* x[y_index(i,j)]
            - x[y_index(i-1,j)] - x[y_index(i+1,j)]
            - x[y_index(i,j-1)] - x[y_index(i,j+1)];

      // Add the forcing term (including the step size here)
      val += hh_*d_cont(h_*1.*i, h_*1.*j,x[y_index(i,j)], x[u_index(i,j)]);
      g[pde_index(i,j)] = val;
    }
  }

  int ig = N_*N_;
  // set up the Neumann boundary conditions
  for (int i=1; i<= N_; i++) {
    g[ig] = (1.+h_*b_i0_)*x[y_index(i,0)] - x[y_index(i,1)];
    ig++;
  }
  for (int i=1; i<= N_; i++) {
    g[ig] = (1.+h_*b_i1_)*x[y_index(i,N_+1)] - x[y_index(i,N_)];
    ig++;
  }
  for (int j=1; j<= N_; j++) {
    g[ig] = (1.+h_*b_0j_)*x[y_index(0,j)] - x[y_index(1,j)];
    ig++;
  }
  for (int j=1; j<= N_; j++) {
    g[ig] = (1.+h_*b_1j_)*x[y_index(N_+1,j)] - x[y_index(N_,j)];
    ig++;
  }  
   
  return true;
}

//*************************************************************************
//
//
//         Nothing has to be changed below this point !!
//
//
//*************************************************************************


bool MyADOLC_sparseNLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
  eval_obj(n,x,obj_value);

  return true;
}

bool MyADOLC_sparseNLP::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{

  gradient(tag_f,n,x,grad_f);

  return true;
}

bool MyADOLC_sparseNLP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{

  eval_constraints(n,x,m,g);

  return true;
}

bool MyADOLC_sparseNLP::eval_jac_g(Index n, const Number* x, bool new_x,
                       Index m, Index nele_jac, Index* iRow, Index *jCol,
                       Number* values)
{

  if (values == NULL) {
    // return the structure of the jacobian

    for(Index idx=0; idx<nnz_jac; idx++)
      {
	iRow[idx] = rind_g[idx];
	jCol[idx] = cind_g[idx];
      }
  }
  else {
    // return the values of the jacobian of the constraints

    sparse_jac(tag_g, m, n, 1, x, &nnz_jac, &rind_g, &cind_g, &jacval, options_g); 

    for(Index idx=0; idx<nnz_jac; idx++)
      {
	values[idx] = jacval[idx];

      }
  }
  return true;
}

bool MyADOLC_sparseNLP::eval_h(Index n, const Number* x, bool new_x,
                   Number obj_factor, Index m, const Number* lambda,
                   bool new_lambda, Index nele_hess, Index* iRow,
                   Index* jCol, Number* values)
{

  if (values == NULL) {
    // return the structure. This is a symmetric matrix, fill the lower left
    // triangle only.

    for(Index idx=0; idx<nnz_L; idx++)
      {
	iRow[idx] = rind_L[idx];
	jCol[idx] = cind_L[idx];
      }
  }
  else {
    // return the values. This is a symmetric matrix, fill the lower left
    // triangle only

    obj_lam[0] = obj_factor;
    for(Index idx = 0; idx<m ; idx++)
      obj_lam[1+idx] = lambda[idx];

    set_param_vec(tag_L, m+1, obj_lam);
    sparse_hess(tag_L, n, 1, const_cast<double*>(x), &nnz_L, &rind_L, &cind_L, &hessval, options_L);
     
    for(Index idx = 0; idx < nnz_L ; idx++)
      {
          values[idx] = hessval[idx];
      }
  }

  return true;
}

void MyADOLC_sparseNLP::finalize_solution(SolverReturn status,
                              Index n, const Number* x, const Number* z_L, const Number* z_U,
                              Index m, const Number* g, const Number* lambda,
                              Number obj_value,
			      const IpoptData* ip_data,
			      IpoptCalculatedQuantities* ip_cq)
{

  printf("\n\nObjective value\n");
  printf("f(x*) = %e\n", obj_value);

// memory deallocation of ADOL-C variables

  delete[] obj_lam;
  delete[] y_d_;
  free(rind_g);
  free(cind_g);
  free(rind_L);
  free(cind_L);
  free(jacval);
  free(hessval);

}


//***************    ADOL-C part ***********************************

void MyADOLC_sparseNLP::generate_tapes(Index n, Index m, Index& nnz_jac_g, Index& nnz_h_lag)
{
  Number *xp    = new double[n];
  Number *lamp  = new double[m];
  Number *zl    = new double[m];
  Number *zu    = new double[m];

  adouble *xa   = new adouble[n];
  adouble *g    = new adouble[m];
  double *lam   = new double[m];
  double sig;
  adouble obj_value;
  
  double dummy;

  int i,j,k,l,ii;

  obj_lam   = new double[m+1];

  get_starting_point(n, 1, xp, 0, zl, zu, m, 0, lamp);

  trace_on(tag_f);
    
    for(Index idx=0;idx<n;idx++)
      xa[idx] <<= xp[idx];

    eval_obj(n,xa,obj_value);

    obj_value >>= dummy;

  trace_off();
  
  trace_on(tag_g);
    
    for(Index idx=0;idx<n;idx++)
      xa[idx] <<= xp[idx];

    eval_constraints(n,xa,m,g);


    for(Index idx=0;idx<m;idx++)
      g[idx] >>= dummy;

  trace_off();

  trace_on(tag_L);
    
    for(Index idx=0;idx<n;idx++)
      xa[idx] <<= xp[idx];
    for(Index idx=0;idx<m;idx++)
      lam[idx] = 1.0;
    sig = 1.0;

    eval_obj(n,xa,obj_value);

    obj_value *= mkparam(sig);
    eval_constraints(n,xa,m,g);
 
    for(Index idx=0;idx<m;idx++)
      obj_value += g[idx]*mkparam(lam[idx]);

    obj_value >>= dummy;

  trace_off();

  rind_g = NULL; 
  cind_g = NULL;
  rind_L = NULL;
  cind_L = NULL;

  options_g[0] = 0;          /* sparsity pattern by index domains (default) */ 
  options_g[1] = 0;          /*                         safe mode (default) */ 
  options_g[2] = 0;
  options_g[3] = 0;          /*                column compression (default) */ 
  
  jacval=NULL;
  hessval=NULL;
  sparse_jac(tag_g, m, n, 0, xp, &nnz_jac, &rind_g, &cind_g, &jacval, options_g); 

  nnz_jac_g = nnz_jac;

  options_L[0] = 0;         
  options_L[1] = 1;        

  sparse_hess(tag_L, n, 0, xp, &nnz_L, &rind_L, &cind_L, &hessval, options_L);
  nnz_h_lag = nnz_L;

  delete[] lam;
  delete[] g;
  delete[] xa;
  delete[] zu;
  delete[] zl;
  delete[] lamp;
  delete[] xp;
}
