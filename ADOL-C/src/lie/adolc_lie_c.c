/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     lie/adolc_lie_c.c
 Revision: $Id$
 Contents: Implementation of functions for computation of Lie derivatives
 

 Copyright (c) Siquian Wang, Klaus Röbenack, Jan Winkler, Mirko Franke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
  
----------------------------------------------------------------------------*/
#include <adolc/adolc.h>
#include <adolc/adouble.h>
#include <adolc/lie/drivers.h>
#include "dvlparms.h"




/** Computes the total derivative of the output
 *
 *  @param p Number of rows of C (number of outputs)
 *  @param n Number of columns of B (number of inputs)
 *  @param deg Order of derivative (d)
 *  @retval ***B Total derivative dx(j+1)/dx0
 *  @param ***C Partial derivative of the output mapping
 *  @param ***D Total derivative of the output mapping
 */
void accodeout (int p, int n, int deg, double ***B,	double ***C, double ***D)
{
	int i, j, k, l, ip, jp, kp;

	// D_k:=C_k (1. summation step)
	for (k=0; k <= deg; k++)
	{
		for (j=0; j < p; j++)
			for (i=0; i < n; i++)
				D[j][i][k]=C[j][i][k];
		
		// add sum if necessary
		if (k >= 1)
		{
			for (l=1; l<=k; l++)
				for (jp=0; jp<p; jp++)
					for (ip=0; ip<n; ip++)
					{
						double x=0.0;
						for (kp=0; kp<n; kp++)
						{
							x+=C[jp][kp][k-l]*B[kp][ip][l-1];
						};
						D[jp][ip][k]+=x;
					};
		};
	};
};

	

/** Helper function for calculation of the Lie derivatives of a covector field
 *
 *  @param n Number of rows = number of columns
 *  @param d Order of derivative (d)
 *  @param ***B Total derivative dx(j+1)/dx0 (B[n][n][d])
 *  @param **C Taylor coefficients of the output mapping (C[n][d])
 *  @retval **D Lie derivative along covector field (D[n][d])
 */
void acccov(int n, int d, double ***B, double **C, double **D)
{
	int i, k, l, ip, kp;

	// factorial      
	int Fak = 1;
	
	// D_k:=Fak*C_k (1. summation step)
	for (k = 0; k <= d; k++)
	{
		if(k == 0)
			Fak = 1;
		else
			Fak = Fak*k;

		for (i = 0; i < n; i++)
			D[i][k]=Fak*C[i][k];

		// add sum if necessary
		if (k>=1)
		{
			double x;
			for (l = 1; l <= k; l++)
			{
				for (ip = 0; ip < n; ip++)
				{
					x = 0.0;
					for (kp = 0; kp < n; kp++)
					{
						x+=C[kp][k-l]*B[kp][ip][l-1];
					};
					D[ip][k]+= Fak*x;
				};
			};
		};
	};
};

	


/** Helper function for calculation of Lie-brackets (solution of the adjoint variational equation)
 *
 *  @param n Number of columns of B (number of inputs)
 *  @param deg Order of derivative
 *  @param ***A Total derivative of A
 *  @retval ***Bs Solution of adjoint variational equation
 */
void accadj(int n, int deg, double ***A, double ***Bs)
{
	int i, j, k, l, ip, jp, kp;

	// (1. summation step)
	for (k = 0; k <= deg; k++)
	{
		for (j = 0; j < n; j++)
		{
			for (i = 0; i < n; i++)
			{
				Bs[j][i][k] = -A[i][j][k]/(k+1);
			}
		}
			
		// add sum if necessary
		if (k >= 1)
		{
			double x = 0.0;
			for (l = 1; l <= k; l++)
			{
				for (jp = 0; jp < n; jp++)
				{
					for (ip = 0; ip < n; ip++)
					{
						x = 0.0;
						for (kp = 0; kp < n; kp++)
						{
							x+=A[kp][jp][k-l]*Bs[kp][ip][l-1];
						};
						Bs[jp][ip][k] -= x/(k+1);
					};
				};
			};
		};
	};
};



/** Calculates the Lie-derivative along a co vector field
 *
 * @param n Number of rows and columns
 * @param d Order of derivative (d)
 * @param Bs Solution of adjoint variational equation (Bs[n][n][d])
 * @param b Taylor-coefficients of output mapping (b[n][d])
 * @retval result Lie derivative along co-vector field (result[n][d])
 */
void accbrac(int n,	int d, double ***Bs, double **b, double **result) 
{
	int i, j, k, l, jp, kp;

	// factorial      
	int Fak = 1;

	// formula 3.58
	for (k = 0; k <= d; k++)
	{
		if(k == 0)
			Fak = 1;
		else
			Fak = Fak*k;

		for (j = 0; j < n; j++)
		{
			for (i = 0; i < n; i++)
			{
				result[i][k] = Fak*b[i][k];
			}

			if(k >= 1)
			{
				double x;
				for (l = 1; l <= k; l++)
				{
					for (jp = 0; jp < n; jp++)
					{
						x = 0.0;
						for (kp = 0; kp < n; kp++)
						{
							x+=Bs[kp][jp][l-1]*b[kp][k-l];
						}
						result[jp][k]+=Fak*x;
					};
				};
			};
		};
	};
};

	




/** Computes the Lie derivative of smooth map h : D -> R^m along a vector field f : D -> R^n
 * \param Tape_F tape identification of vector field f
 * \param Tape_H tape identification of vector field h
 * \param n      number of independent variables n
 * \param m	     number of dependent variables m
 * \param x0     values of independent variables x0 (dimension [n])
 * \param d      highest derivative degree d
 * \param result resulting Lie derivatives of vectorial scalar fields (dimension [m][d+1])
 */
int lie_scalarcv(short Tape_F, short Tape_H, short n, short m, double* x0, short d,	double** result) 
{
	double** X = myalloc2(n, d+1);   // Taylorcoeff. expansion x(t)
	double** Y = myalloc2(m, d+1);   // Taylorcoeff. expansion y(t)
	double*  x = myalloc1(n);
	double*  y = myalloc1(m);
	int i=0, j=0, k=0;
	double Fak = 1.0;

	for (i = 0; i < n; i++) 
	{
		X[i][0] = x0[i];
	};
	
	//see odedrivers
	forodec(Tape_F, n, 1.0, 0, d, X);

	//prepare for input
	for (i = 0; i < n; i++) 
	{
		x[i] = X[i][0];
		for (k = 0; k < d; k++)
			X[i][k] = X[i][k+1];
	}

	hos_forward(Tape_H, m, n, d, 0, x, X, y, Y);

	//postprocess output of hos_forward
	for (i=0; i<m; i++) 
	{
		for (k = d; k > 0; k--)
		{
			Y[i][k] = Y[i][k-1];
		};
		Y[i][0] = y[i];
	};

	// prepare output for lie_Scalar
	for(j=0;j<m;j++)
	{
		for (i = 0; i <= d; i++)
		{
			result[j][i] = Fak*Y[j][i];
			Fak = Fak*(i+1);
		}
		Fak=1.0;
	}

	myfree2(X);
	myfree2(Y);
	myfree1(x);
	myfree1(y);

	return -1;
}


/** Lie derivative of scalar field h : D -> R^m along vector field f : D -> R^n
 *  \param Tape_F tape identification of vector field f
 *  \param Tape_H tape identification of scalar field h
 *  \param n      number of independent variables n and m = 1
 *  \param x0     values of independent variables x0 (dimension [n])
 *  \param d	  highest derivative degree d
 *  \retval result resulting Lie derivatives of a scalar field (dimension [d+1])
 */
int lie_scalarc(short Tape_F, short Tape_H, short n, double* x0, short d, double* result) 
{
	int rc= -1;
	short m = 1, i=0;
	double** Temp = myalloc2(m, d+1);

	rc = lie_scalarcv(Tape_F, Tape_H, n, m, x0, d, Temp);

	for (i = 0; i <= d; i++)
	{
		result[i] = Temp[0][i];
	}

	myfree2(Temp);      

	return rc;
}


/** Calculates the jacobians of the Lie derivatives of scalar fields h : D -> R^m
 *  \param Tape_F tape identification of vector field f
 *  \param Tape_H tape identification of vector field h
 *  \param n      number of independent variables n
 *  \param m      number of dependent variables m
 *  \param x0     values of independent variables x0 (dimension [n])
 *  \param d      highest derivative degree d
 *  \retval result resulting jacobians of Lie derivatives of vectorial scalar fields (dimension [m][n][d+1])
 */
int lie_gradientcv(short Tape_F, short Tape_H, short n,	short m, double* x0, short d, double*** result) 
{
	double **X=myalloc2(n,d+1);
	double **Y=myalloc2(m,d+1);
	double ***Pc=myalloc3(m,n,d+1);
	double ***A=myalloc3(n,n,d);
	double ***B=myalloc3(n,n,d);
	double ***D=myalloc3(m,n,d+1);

	double* x = myalloc1(n);
	double* y = myalloc1(m);
	double* xp = myalloc1(n);
	double* yp = myalloc1(m);

	static int depax_m,depax_n;
	static double** In;
	static double** Im;

	int i=0, j=0, l=0, k=0, rc=-1;
	double Fak=1.0;

	for (i = 0; i < n; i++) 
		X[i][0] = x0[i];

	forodec(Tape_F, n, 1.0, 0, d, X);

	if (n adolc_compsize depax_n) {
		if (depax_n)
			myfreeI2(depax_n,In);
		In = myallocI2(depax_n = n);
	}
	if (m adolc_compsize depax_m) {
		if (depax_m)
			myfreeI2(depax_m,Im);
		Im = myallocI2(depax_m = m);
	}

	hov_reverse(Tape_F,n,n,d-1,n,In,A,0);// explanation in interfaces.cpp
	accodec(n, 1.0, d-1, A, B, 0);		 // explanation in odedrivers.c

	//prepare for input
	for (i=0; i<n; i++) 
	{
		x[i] = X[i][0];
		if (d == 1)
		{
			xp[i] = X[i][1];
		}
		else
		{
			for (k=0; k<d; k++)
			{
				X[i][k] = X[i][k+1];
			}
		}
	}

	hos_forward(Tape_H, m, n, d, d+1, x, X, y, Y);

	hov_reverse(Tape_H, m, n, d, m, Im, Pc, 0);  
	accodeout(m, n, d, B, Pc, D);

	for(l=0; l<m; l++)
	{
		for(i=0; i<n; i++)
		{
			Fak=1.0;
			for(j=0; j<=d; j++)
			{
				result[l][i][j]=Fak*D[l][i][j];
				Fak=Fak*(j+1);
			}
		}
	}

	myfree2(X);
	myfree2(Y);
	myfree3(Pc);
	myfree3(A);
	myfree3(B);
	myfree3(D);

	myfree1(x);
	myfree1(y);
	myfree1(xp);
	myfree1(yp);

	return rc;
}




/** Computes the gradients of the Lie derivatives of a scalar field h : D -> R
 *
 *  \param Tape_F tape identification of vector field f
 *  \param Tape_H tape identification of vector field h
 *  \param n      number of independent variables n
 *  \param x0     values of independent variables x0 (dimension [n])
 *  \param d      highest derivative degree d
 *  \retval result resulting jacobians of Lie derivatives of a scalar field (dimension [n][d+1])
 */
int lie_gradientc(short Tape_F,	short Tape_H, short n, double* x0, short d,	double** result) 
{
	int rc= -1;
	short m = 1, i=0, j=0;
	double*** Temp = myalloc3(m, n, d+1);

	rc = lie_gradientcv(Tape_F, Tape_H, n, m, x0, d, Temp);

	for(i=0; i<n; i++)
		for(j=0; j<=d; j++)
		{
			result[i][j]=Temp[0][i][j];

		}

		myfree3(Temp);

		return rc;
}

	

/** Computes the Lie derivatives of the covector field w : D -> (R^m)* along the vector field f : D -> R^n
 *
 *  \param Tape_F tape identification of vector field f
 *  \param Tape_W tape identification of covector field h
 *  \param n      number of independent variables n
 *  \param x0     values of independent variables x0 (dimension [n])
 *  \param d      highest derivative degree d
 *  \retval result resulting Lie derivatives of a covector field (dimension [n][d+1])
 */
int lie_covector( short int Tape_F, short int Tape_W, short int n, double* x0, short int d, double** result)
{
	int m=n;                   
	double** X = myalloc2(n, d+1);   // Taylorcoeff. expansion x(t)
	double** Y = myalloc2(m, d+1);   // Taylorcoeff. expansion y(t)

	double***A = myalloc3(n, n, d);    
	double***B = myalloc3(n, n, d+1);

	double* x = myalloc1(n);
	double* y = myalloc1(m);
	double* xp = myalloc1(n);
	double* yp = myalloc1(m);

	int i=0, k=0;

	static int depax_m,depax_n;
	static double** In;
	static double** Im;

	for (i = 0; i < n; i++) X[i][0] = x0[i];

	forodec(Tape_F, n, 1.0, 0, d, X);

	if (n adolc_compsize depax_n) {
		if (depax_n)
			myfreeI2(depax_n,In);
		In = myallocI2(depax_n = n);
	}
	if (m adolc_compsize depax_m) {
		if (depax_m)
			myfreeI2(depax_m,Im);
		Im = myallocI2(depax_m = m);
	}

	hov_reverse(Tape_F,n,n,d-1,n,In,A,0);	// explanation in interfaces.cpp

	//prepare for input
	for (i=0; i<n; i++) {
		x[i] = X[i][0];
		if (d == 1)
			xp[i] = X[i][1];
		else
			for (k=0; k<d; k++)
				X[i][k] = X[i][k+1];
	}

	hos_forward(Tape_W, m, n, d, d+1, x, X, y, Y);

	//postprocess output of hos_forward
	for (i=0; i<m; i++) {
		if (d == 1)
			Y[i][1] = yp[i];
		else
			for (k=d; k>0; k--)
				Y[i][k] = Y[i][k-1];
		Y[i][0] = y[i];
	}

	accodec(n, 1.0, d-1, A, B, 0);      // explanation in odedrivers.c
	acccov(n, d, B, Y, result);  

	myfree2(X);
	myfree2(Y);
	myfree3(A);
	myfree3(B);

	myfree1(x);
	myfree1(y);
	myfree1(xp);
	myfree1(yp);

	return -1;
}




/** Calculates the iterated Lie derivatives (Lie brackets) of the vector field g : D -> R^n along the vector field f : D -> R^n
 *
 *  \param Tape_F tape identification of vector field f
 *  \param Tape_G tape identification of vector field g
 *  \param n      number of independent variables n
 *  \param x0     values of independent variables x0 (dimension [n])
 *  \param d      highest derivative degree d
 */
int lie_bracket(short int Tape_F, short int Tape_G, short int n, double* x0, short int d, double** result)
{
	int m = n;                     
	double  **X  = myalloc2(n, d+2);   // Taylorcoeff. expansion x(t)
	double  **Y  = myalloc2(m, d+2);   // Taylorcoeff. expansion y(t)
	double ***A  = myalloc3(n, n, d+1);    
	double ***Xs = myalloc3(n, n, d+1);

	double* y = myalloc1(m);

	int i, k;

	//static identity matrix for hov_reverse
	static int      depax_n = 0;
	static double** In      = NULL;


	for (i = 0; i < n; i++) 
	{
		X[i][0] = x0[i];
	};
	forodec(Tape_F, n, 1.0, 0, d+1, X);

	// for hov_reverse
	if (n > depax_n) 
	{
		if (depax_n)
		{
			myfreeI2(depax_n, In);
		};
		In = myallocI2(depax_n = n);
	}
	hov_reverse(Tape_F, n, n, d, n, In, A, 0);

	//prepare for input
	for (i = 0; i < n; i++) 
	{
		for (k = 0; k < d; k++)
		{
			X[i][k] = X[i][k+1];
		}
	}
	hos_forward(Tape_G, m, n, d+1, d+2, x0, X, y, Y);

	//postprocess output of hos_forward
	for (i = 0; i < m; i++) 
	{
		for (k = d; k > 0; k--)
		{
			Y[i][k] = Y[i][k-1];
		}
		Y[i][0] = y[i];
	}

	accadj(n, d, A, Xs); 
	accbrac(n, d, Xs, Y, result);  

	myfree1(y);
	myfree2(X);
	myfree2(Y);
	myfree3(A);
	myfree3(Xs);
	
	return -1;
};


