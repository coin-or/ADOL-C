/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     traceless_cuda.cu
 Revision: $Id$
 Contents: computation of coordinate transform, 
           cuda example described in the manual

 Copyright (c) Andrea Walther, Alina Koniaeva
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */

#include "adoublecuda.h"
#include <cuda.h>
#include <iostream>

using namespace std;

__global__ void kernel(double* inx, double* outy, double* outderiv) {

    const int index = threadIdx.x ;
    const int index1 = threadIdx.y;
    const int index2 = blockIdx.x;
    const int dim = blockDim.x*blockDim.y;
    const int index3 = blockDim.x;

    // Declare dependent and independent variables as adoubles
    adtlc::adouble y[3];
    adtlc::adouble x[3];

    // Read out point for function evaluation
    for(int i=0;i<3;i++)
	x[i]=inx[index2*dim+index*3+i];

    // Set direction for calculation of derivatives
    x[index1].setADValue(1);

    // Function evaluation
    y[0] = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    y[1] = atan(sqrt(x[0]*x[0]+x[1]*x[1])/x[2]);
    y[2] = atan(x[1]/x[0]);

   for(int i=0; i<3; i++) outy[(index2*index3+index)*3+i]=y[i].getValue();
   for(int i=0; i<3; i++) outderiv[(index2*dim+index*3+index1)*3+i]=y[i].getADValue();
	
}
  
cudaError_t kernellaunch(double* inx, double* outy, double* outderiv, int n) {
    // Create 16 blocks
    int Blocks=16;
    // Two dimensional (M/Blocks) x 3 blocks
    dim3 threadsPerBlock(n/Blocks,3);
    
    // Call kernel function with 16 blocks with (M/Blocks) x 3 threads per block
    kernel <<< Blocks, threadsPerBlock >>>( inx, outy, outderiv);
    cudaError_t cudaErr = cudaGetLastError();

    return cudaErr;
}


int main(){

    int M=1024;
    double* deriv = new double[9*M];
    double* y = new double[3*M];
    double* x = new double[3*M];

    // Initialize x_i
    for(int k=0; k < M; k++){
     for (int i=0; i<3; ++i)          
	x[k*3+i] =i + 1/(k+1);}

    // Allocate array for independent and dependent variables and Jacobian matrices on GPU
    double * devx;
    cudaMalloc((void**)&devx, 3*M*sizeof(double));

    double * devy;
    cudaMalloc((void**)&devy, 3*M*sizeof(double));

    double * devderiv;
    cudaMalloc((void**)&devderiv, 3*3*M*sizeof(double));

    // Copy values of independent variables from host to GPU
    cudaMemcpy(devx, x, sizeof(double)*3*M, cudaMemcpyHostToDevice);
 
    // Call function to specify amount of blocks and threads to be used
    kernellaunch(devx, devy, devderiv,M);
  
    // Copy values of dependent variables and Jacobian matrices from GPU to host
    cudaMemcpy(y, devy, sizeof(double)*3*M, cudaMemcpyDeviceToHost);

    cudaMemcpy(deriv, devderiv, sizeof(double)*M*3*3, cudaMemcpyDeviceToHost);
}
