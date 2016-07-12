/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     liborpar.cpp
 Revision: $Id$
 Contents: example for differentiation of OpemMP parallel programs
           parallel version 

 Copyright (c) Andrea Walther
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/* Program to compute deltas and vegas of swaption portfolio
   from forward and reverse mode pathwise sensitivities 
   in parallel written by Andrea Walther in 2008-11 based on 
   code written by written by Mike Giles in 2005-7 which is 
   again based on code written by Zhao and Glasserman at 
   Columbia University  */

using namespace std;

#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <cmath>

#include "adolc/adolc.h"

#ifdef _OPENMP
#include <omp.h>
#include <adolc/adolc_openmp.h>
#endif

/* calculate path values */

template <typename ADdouble>
void path_calc(const int N, const int Nmat, const double delta,
	       ADdouble L[], const double lambda[], ADdouble z[])
{
  int      i, n;
  double   lam, con1;
  ADdouble v, vrat;
  ADdouble sqez;
  
  for(n=0; n<Nmat; n++) {
    sqez = sqrt(delta)*z[n];
    
    v = 0.0;
    for (i=n+1; i<N; i++) {
      lam  = lambda[i-n-1];
      con1 = delta*lam;
      v   += (con1*L[i])/(1.0+delta*L[i]);
      vrat = exp(con1*v + lam*(sqez-0.5*con1));
      L[i] = L[i]*vrat;
    }
  }
}

/* calculate the portfolio value v */

template <typename ADdouble>
void portfolio(const int N, const int Nmat, const double delta,
	       const int Nopt, const int maturities[],
	       const double swaprates[],
	       const ADdouble L[], ADdouble& v )
{
  int      i, m, n;
  ADdouble b, s, swapval, *B, *S;
  B = new ADdouble[N];
  S = new ADdouble[N];
  
  b = 1.0;
  s = 0.0;
  
  for (n=Nmat; n<N; n++) {
    b    = b/(1.0+delta*L[n]);
    s    = s + delta*b;
    B[n] = b;
    S[n] = s;
  }
  
  v = 0;
  
  for (i=0; i<Nopt; i++){
    m = maturities[i] + Nmat-1;
    swapval = B[m] + swaprates[i]*S[m] - 1.0;
    condassign(v,-swapval,v-100.0*swapval);
  }
  
  // apply discount //
  
  for (n=0; n<Nmat; n++)
    v = v/(1.0+delta*L[n]);
  delete[](B); 
  delete[](S);  
}

/* -------------------------------------------------------- */

int main(){

  // constants for all threads

  // LIBOR interval  //
  double delta = 0.25; 
  // data for swaption portfolio //
  int    Nopt = 15;
  int    maturities[] = {4,4,4,8,8,8,20,20,20,28,28,28,40,40,40};
  double swaprates[]  = {.045,.05,.055,.045,.05,.055,.045,.05,
                         .055,.045,.05,.055,.045,.05,.055 };

  int       i, j, N, Nmat, npath, nthreads, slot;
  double    vtot, *v, *lambda, **z, **grad, *gradtot, **xp;

  Nmat = 40;
  N = Nmat+40;
  npath = 30;
  nthreads = 5;
  slot = npath/nthreads;

  lambda   = new double[N];
  v        = new double[npath];
  gradtot  = new double[N];
  z        = new double*[npath];
  grad     = new double*[npath];
  xp       = new double*[npath];
  for (i=0;i<npath;i++) 
    {
      z[i] = new double[N];
      grad[i] = new double[N+Nmat];
      xp[i] = new double[N+Nmat];
   }

  for (i=0;i<N;i++) 
    {
      gradtot[i] = 0.0;
      lambda[i] = 0.2;
    }

  for (j=0; j<npath; j++)
    {
      v[j] = 0;
      for (i=0; i<N; i++) 
	  xp[j][i]=  0.05;
      for (i=0; i<Nmat; i++) 
	{
	  z[j][i] = 0.3;
	  xp[j][N+i]=  0.3;
	}
    }

  omp_set_num_threads(nthreads);

  //----------------------------------------------------------//
  //                                                          //
  // do a full path + portfolio sensitivity check             //
  //                                                          //
  // A real application would generate a different random     //
  // vector z for each path but here we set one and reuse it  //
  //                                                          //
  //----------------------------------------------------------//

#pragma omp parallel firstprivate(ADOLC_OpenMP_Handler)
  {    
    // different paths for each thread
    int index = omp_get_thread_num();
    int l,k;
    int rv = 0;
    
    adouble *La, va, *za;

    La = new adouble[N];
    za  = new adouble[Nmat];

    int init = index*slot;

    trace_on(init);
      for(k=0;k<N;k++) 
	La[k] <<= 0.050000;
      for(j=0;j<Nmat;j++) 
        za[j] <<= z[init][j];
    
      path_calc(N,Nmat,delta,La,lambda,za);
      portfolio(N,Nmat,delta,Nopt,maturities,swaprates,La,va);
	
      va >>= v[init];
    trace_off(1);
    
    for(l=init;l<init+slot;l++)
      {
	rv = gradient(init,N+Nmat,xp[l],grad[l]);
      }

#pragma omp barrier
  if (index==0)
    {
      vtot = 0;
      for (l=0; l<npath; l++)
	{
	  vtot += v[l];
	  for(k=0;k<N;k++)
	    gradtot[k] += grad[l][k];
	}
      vtot = vtot/npath;
      for(k=0;k<N;k++)
	gradtot[k] /= npath;

      printf("Gradient: \n");
      for(i=0;i<N;i++)
	printf(" %f \n",gradtot[i]);
    }
  }  


  return 0;
}

