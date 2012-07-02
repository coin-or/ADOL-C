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

    for(Index idx = 0; idx<n ; idx++)
      x_lam[idx] = x[idx];
    for(Index idx = 0; idx<m ; idx++)
      x_lam[n+idx] = lambda[idx];
    x_lam[n+m] = obj_factor;

    sparse_hess(tag_L, n+m+1, 1, x_lam, &nnz_L_total, &rind_L_total, &cind_L_total, &hessval, options_L);
     
    Index idx = 0;
    for(Index idx_total = 0; idx_total <nnz_L_total ; idx_total++)
      {
	if((rind_L_total[idx_total] < (unsigned int) n) && (cind_L_total[idx_total] < (unsigned int) n))
	  {
	    values[idx] = hessval[idx_total];
	    idx++;
	  }
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

  delete x_lam;
  delete rind_g;
  delete cind_g;
  delete rind_L;
  delete cind_L;
  delete rind_L_total;
  delete cind_L_total;
  delete jacval;
  delete hessval;

  for (int i=0;i<n+m+1;i++) {
     free(HP_t[i]);
   }
  free(HP_t);
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
  adouble *lam  = new adouble[m];
  adouble sig;
  adouble obj_value;
  
  double dummy;
  double *jacval;

  int i,j,k,l,ii;

  x_lam   = new double[n+m+1];

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
      lam[idx] <<= 1.0;
    sig <<= 1.0;

    eval_obj(n,xa,obj_value);

    obj_value *= sig;
    eval_constraints(n,xa,m,g);
 
    for(Index idx=0;idx<m;idx++)
      obj_value += g[idx]*lam[idx];

    obj_value >>= dummy;

  trace_off();

  rind_g = NULL; 
  cind_g = NULL;

  options_g[0] = 0;          /* sparsity pattern by index domains (default) */ 
  options_g[1] = 0;          /*                         safe mode (default) */ 
  options_g[2] = -1;         /*                     &jacval is not computed */ 
  options_g[3] = 0;          /*                column compression (default) */ 
  
  jacval=NULL;
  sparse_jac(tag_g, m, n, 0, xp, &nnz_jac, &rind_g, &cind_g, &jacval, options_g); 

  options_g[2] = 0;
  nnz_jac_g = nnz_jac;

  unsigned int  **JP_f=NULL;                /* compressed block row storage */
  unsigned int  **JP_g=NULL;                /* compressed block row storage */
  unsigned int  **HP_f=NULL;                /* compressed block row storage */
  unsigned int  **HP_g=NULL;                /* compressed block row storage */
  unsigned int  *HP_length=NULL;            /* length of arrays */
  unsigned int  *temp=NULL;                 /* help array */

  int ctrl_H;

  JP_f = (unsigned int **) malloc(sizeof(unsigned int*));
  JP_g = (unsigned int **) malloc(m*sizeof(unsigned int*));
  HP_f = (unsigned int **) malloc(n*sizeof(unsigned int*)); 
  HP_g = (unsigned int **) malloc(n*sizeof(unsigned int*));
  HP_t = (unsigned int **) malloc((n+m+1)*sizeof(unsigned int*));
  HP_length = (unsigned int *) malloc((n)*sizeof(unsigned int));
  ctrl_H = 0;

  hess_pat(tag_f, n, xp, HP_f, ctrl_H);

  indopro_forward_safe(tag_f, 1, n, xp, JP_f);
  indopro_forward_safe(tag_g, m, n, xp, JP_g);
  nonl_ind_forward_safe(tag_g, m, n, xp, HP_g);

  for (i=0;i<n;i++) 
    {
      if (HP_f[i][0]+HP_g[i][0]!=0)
	{
	  if (HP_f[i][0]==0)
	    {
	      HP_t[i] = (unsigned int *) malloc((HP_g[i][0]+HPOFF)*sizeof(unsigned int));
	      for(j=0;j<=(int) HP_g[i][0];j++)
		{
		  HP_t[i][j] = HP_g[i][j];
		}
	      HP_length[i] = HP_g[i][0]+HPOFF;
	    }
	  else
	    {
	      if (HP_g[i][0]==0)
		{
		  HP_t[i] = (unsigned int *) malloc((HP_f[i][0]+HPOFF)*sizeof(unsigned int));
		  for(j=0;j<=(int) HP_f[i][0];j++)
		    {
		      HP_t[i][j] = HP_f[i][j];
		    }
		  HP_length[i] = HP_f[i][0]+HPOFF;
		}
	      else
		{
		  HP_t[i] = (unsigned int *) malloc((HP_f[i][0]+HP_g[i][0]+HPOFF)*sizeof(unsigned int));
		  k = l = j = 1;
		  while ((k<=(int) HP_f[i][0]) && (l <= (int) HP_g[i][0]))
		    {
		      if (HP_f[i][k] < HP_g[i][l])
			{
			  HP_t[i][j]=HP_f[i][k];
			  j++; k++;
			}
		      else
			{
			  if (HP_f[i][k] == HP_g[i][l])
			    {
			      HP_t[i][j]=HP_f[i][k];
			      l++;j++;k++;
			    }
			  else
			    {
			      HP_t[i][j]=HP_g[i][l];
			      j++;l++;		      
			    }
			}
		    } // end while
		  for(ii=k;ii<=(int) HP_f[i][0];ii++)
		    {
		      HP_t[i][j] = HP_f[i][ii];
		      j++;
		    }
		  for(ii=l;ii<=(int) HP_g[i][0];ii++)
		    {
		      HP_t[i][j] = HP_g[i][ii];
		      j++;
		    }
		  
		}
	    }
	  HP_t[i][0]=j-1;
	  HP_length[i] = HP_f[i][0]+HP_g[i][0]+HPOFF;
	}
      else
	{
	  HP_t[i] = (unsigned int *) malloc((HPOFF+1)*sizeof(unsigned int));
	  HP_t[i][0]=0;
	  HP_length[i]=HPOFF;
	}
    }   

  for (i=0;i<m;i++) 
    {
      HP_t[n+i] = (unsigned int *) malloc((JP_g[i][0]+1)*sizeof(unsigned int));
      HP_t[n+i][0]=JP_g[i][0];
      for(j=1;j<= (int) JP_g[i][0];j++)
	{
	  HP_t[n+i][j]=JP_g[i][j];
	  if (HP_length[JP_g[i][j]] < HP_t[JP_g[i][j]][0]+1)
	    {
	      temp = (unsigned int *) malloc((HP_t[JP_g[i][j]][0])*sizeof(unsigned int));
	      for(l=0;l<=(int)HP_t[JP_g[i][j]][0];l++)
		temp[l] = HP_t[JP_g[i][j]][l];
 	      free(HP_t[JP_g[i][j]]);
	      HP_t[JP_g[i][j]] = (unsigned int *) malloc(2*HP_length[JP_g[i][j]]*sizeof(unsigned int));
	      HP_length[JP_g[i][j]] = 2*HP_length[JP_g[i][j]];
	      for(l=0;l<=(int)temp[0];l++)
		HP_t[JP_g[i][j]][l] =temp[l];
	      free(temp);
	    }
 	  HP_t[JP_g[i][j]][0] = HP_t[JP_g[i][j]][0]+1;
 	  HP_t[JP_g[i][j]][HP_t[JP_g[i][j]][0]] = i+n;
	}
    }


  for(j=1;j<= (int) JP_f[0][0];j++)
    {
      if (HP_length[JP_f[0][j]] < HP_t[JP_f[0][j]][0]+1)
	{
	  temp = (unsigned int *) malloc((HP_t[JP_f[0][j]][0])*sizeof(unsigned int));
	  for(l=0;l<=(int)HP_t[JP_f[0][j]][0];l++)
	    temp[l] = HP_t[JP_f[0][j]][l];
	  free(HP_t[JP_f[0][j]]);
	  HP_t[JP_f[0][j]] = (unsigned int *) malloc(2*HP_length[JP_f[0][j]]*sizeof(unsigned int));
	  HP_length[JP_f[0][j]] = 2*HP_length[JP_f[0][j]];
	  for(l=0;l<=(int)temp[0];l++)
	    HP_t[JP_f[0][j]][l] =temp[l];
	  free(temp);
	}
      HP_t[JP_f[0][j]][0] = HP_t[JP_f[0][j]][0]+1;
      HP_t[JP_f[0][j]][HP_t[JP_f[0][j]][0]] = n+m;
    }


  HP_t[n+m] = (unsigned int *) malloc((JP_f[0][0]+2)*sizeof(unsigned int));
  HP_t[n+m][0]=JP_f[0][0]+1;
  for(j=1;j<= (int) JP_f[0][0];j++)
    HP_t[n+m][j]=JP_f[0][j];
  HP_t[n+m][JP_f[0][0]+1]=n+m;



  set_HP(tag_L,n+m+1,HP_t);

  nnz_h_lag = 0;
   for (i=0;i<n;i++) {
    for (j=1;j<=(int) HP_t[i][0];j++)
      if ((int) HP_t[i][j] <= i)
	nnz_h_lag++;
     free(HP_f[i]);
     free(HP_g[i]);
   }
  nnz_L = nnz_h_lag;

  options_L[0] = 0;         
  options_L[1] = 1;        

  sparse_hess(tag_L, n+m+1, -1, xp, &nnz_L_total, &rind_L_total, &cind_L_total, &hessval, options_L);

  rind_L = new unsigned int[nnz_L];
  cind_L = new unsigned int[nnz_L];
  rind_L_total = new unsigned int[nnz_L_total];
  cind_L_total = new unsigned int[nnz_L_total];

  unsigned int ind = 0;

  for (int i=0;i<n;i++) 
    for (unsigned int j=1;j<=HP_t[i][0];j++)
      {
	if (((int) HP_t[i][j]>=i) &&((int) HP_t[i][j]<n)) 
	  {
	    rind_L[ind] = i;
	    cind_L[ind++] = HP_t[i][j];
	  }
      }

   ind = 0;
   for (int i=0;i<n+m+1;i++) 
     for (unsigned int j=1;j<=HP_t[i][0];j++)
       {
	if ((int) HP_t[i][j]>=i) 
	  {
	    rind_L_total[ind] = i;
	    cind_L_total[ind++] = HP_t[i][j];
	  }
       }

  for (i=0;i<m;i++) {
     free(JP_g[i]);
   }

  free(JP_f[0]);
  free(JP_f);
  free(JP_g);
  free(HP_f);
  free(HP_g);
  free(HP_length);

  delete[] lam;
  delete[] g;
  delete[] xa;
  delete[] zu;
  delete[] zl;
  delete[] lamp;
  delete[] xp;
}
