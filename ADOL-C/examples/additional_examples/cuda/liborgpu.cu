 /*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     liborgpu.cu
 Revision: $Id$
 Contents: example for differentiation of GPU parallel programs

 Copyright (c) Andrea Walther, Alina Koniaeva
  
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

#include "adoublecuda.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <cassert>
#include <fstream>
using namespace std;

// LIBOR interval  
#define delta 0.25

// data for swaption portfolio 
#define Nopt 15

#define N 80
#define Nmat 40

//kernel code
__global__ void portfolio_kernel(double* inx, double* outy, double* dery) {
	int i,n,m;

	double lambda[N];
    	for (i=0;i<N;i++) 
    		lambda[i] = 0.2;
	
	//thread index in x-dimension (here always 0)
	const int index = threadIdx.x;
	//thread index in y-dimesion
	const int index1 = threadIdx.y;
	//block index
	const int index2 = blockIdx.x;
	//block size in x-dimension (here always 1)
	const int index3 = blockDim.x;
	//block size (here always N+Nmat)
	const int dim = blockDim.x*blockDim.y;

	adtlc::adouble L[N];
	adtlc::adouble z[Nmat];
	
	//initialize independent variables
	for(i=0; i < N; i++)
		L[i]=inx[index2*dim+index*(N+Nmat)+i];
	for(i=0; i< Nmat; i++)
		z[i]=inx[index2*dim+index*(N+Nmat)+N+i];
	
        //determine which derivative (with respect to which variable) is calculated in a thread
	if(index1<N)
		L[index1].setADValue(1.0);
	else
		z[index1-N].setADValue(1.0);

	/* calculate path values */

  	double   lam, con1;
	adtlc::adouble v, vrat;
	adtlc::adouble sqez;
  
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

	/* calculate the portfolio value v */

	int    maturities[] = {4,4,4,8,8,8,20,20,20,28,28,28,40,40,40};
    	double swaprates[]  = {.045,.05,.055,.045,.05,.055,.045,.05,
                         .055,.045,.05,.055,.045,.05,.055 };

	  adtlc::adouble b, s, y, swapval;
	  adtlc::adouble B[N];
	  adtlc::adouble S[N];
  
	  b = 1.0;
	  s = 0.0;
  
	  for (n=Nmat; n<N; n++) {
	    b    = b/(1.0+delta*L[n]);
	    s    = s + delta*b;
	    B[n] = b;
	    S[n] = s;
	  }
  
  	  y = 0;

  	  for (i=0; i<Nopt; i++){
    		m = maturities[i] + Nmat-1;
    		swapval = B[m] + swaprates[i]*S[m] - 1.0;
    		if(-swapval.getValue() > 0)
			y = y-100.0*swapval;
  	  }

  	  // apply discount //
  
 	  for (n=0; n<Nmat; n++)
    		y = y/(1.0+delta*L[n]);

	  // save results of the function evaluations in an array
	  outy[index2*index3+index]=y.getValue();
	 
          //save results of gradient calculation in an array
	  dery[index2*dim+index*(N+Nmat)+index1]=y.getADValue(); 
}

cudaError_t portfolio(double* inx, double* outy, double* dery, int npath) {
    //two dimensional 1x(N+Nmat) blocks
    dim3 threadsPerBlock(1, N+Nmat);

    //call kernel function with npath Blocks with 1x(N+Nmat) threads per block
    portfolio_kernel <<<npath , threadsPerBlock >>>(inx, outy, dery);
    cudaError_t cudaErr = cudaGetLastError();

    return cudaErr;
}

/* -------------------------------------------------------- */

int main(){

    cudaError_t cudaErr;
        
    int       i, j, npath;
    double    vtot,*v, *lambda, **z,**grad, *gradtot, **xp;

    npath=10;
	
    lambda   = new double[N];
    v        = new double[npath];
    gradtot  = new double[N];
    z        = new double*[npath];
    grad     = new double*[npath];
    xp       = new double*[npath];

    //array for values of the independent variables (on host)	    
    double* host = (double*) malloc((N+Nmat)*npath*sizeof(double));
    
    //array for result of function evaluations (on host)
    double* hostres =(double*) malloc(npath*sizeof(double));

    //array for result of gradient evaluations (on host)
    double* hostder = (double*)malloc((N+Nmat)*npath*sizeof(double));

    //array for independent variables (on GPU)
    double *devx;

    //array for result of function evaluations (on GPU)
    double *devy;

    //array for result of gradient evaluations (on GPU)
    double *dery;

    for (i=0; i < npath; i++) 
   	{
     		z[i] = new double[Nmat];
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

     for(int k=0; k < npath; k++)
      {
		for(int s=0; s < N+Nmat; s++)
			host[k*(N+Nmat)+s]=xp[k][s];
      }

     //allocate array for independent variables on GPU	
     cudaErr = cudaMalloc((void**)&devx, npath*(N+Nmat)*sizeof(double));

     //copy values of independent variables from host to GPU
     cudaErr = cudaMemcpy(devx, host, sizeof(double)*npath*(N+Nmat), cudaMemcpyHostToDevice);

     //allocate array for dependent variables on GPU
     cudaErr = cudaMalloc((void**)&devy, npath*sizeof(double));

     //allocate array for gradient values on GPU
     cudaErr = cudaMalloc((void**)&dery, npath*(N+Nmat)*sizeof(double));
                                                          
     //full path + portfolio sensitivity check            
     portfolio(devx, devy, dery, npath);

     //copy values of dependent variables from GPU to host
     cudaErr = cudaMemcpy(hostres, devy, npath*sizeof(double), cudaMemcpyDeviceToHost);

     //copy values of gradients from GPU to host
     cudaErr = cudaMemcpy(hostder, dery, npath*(N+Nmat)*sizeof(double), cudaMemcpyDeviceToHost);

     for(i=0; i<npath ; i++)
	{
		for(j=0; j<N+Nmat ;j++)
			grad[i][j]=hostder[i*(N+Nmat)+j];
	}

     vtot = 0;
     for (i=0; i<npath; i++)
      {
	vtot += v[i];
	for(j=0;j<N;j++)
		gradtot[j] += grad[i][j];
      }
     vtot = vtot/npath;
     for(j=0;j<N;j++)
     	gradtot[j] /= npath;

     printf("Gradient: \n");
     for(i=0;i<N;i++)
      printf(" %f \n",gradtot[i]);

     delete[] lambda;
     delete[] gradtot;
     delete[] z;
     delete[] grad;
     delete[] xp;
     delete[] host;
     delete[] hostres;
     delete[] hostder;

}
