/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     GantryCrane.cpp
 Revision: $Id$
 Contents: example for calculation of Lie derivatives
  
  
 Copyright (c) Siquian Wang, Klaus Röbenack, Jan Winkler, Mirko Franke
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/** By the example of the gantry crane (as shown in Röbenack, Winkler and 
 *  Wang. 'LIEDRIVERS - A Toolbox for the Efficient Computation of Lie 
 *  Derivatives Based on the Object-Oriented Algorithmic Differentiation 
 *  Package ADOL-C') the usage of the drivers of the Lie Toolbox is 
 *  illustrated. 
 *  Beside Lie derivatives of scalar fields and their gradients also Lie
 *  brackets are computed.
 */

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>
#include <adolc/lie/drivers.h>
#include <iostream>

/****************************************************************************/
/*                                                   NAMESPACES AND DEFINES */
using namespace std;

#define TAPE_F 1
#define TAPE_G 2
#define TAPE_H 3

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main()
{
	const int n = 4, m_H = 2;
	double* x0 = myalloc(n);
	double vf[n], vg[n], vh[n];
	adouble aX[n], af[n], ag[n], ah[m_H];
	const double mc = 1.0, ml = 1.0, l  = 1.0, g  = 9.81;


	/****************************
	 * Trace for vector field f *
	 ****************************/
	trace_on(TAPE_F);
	{
		for (int i = 0; i < n; i++)
			aX[i] <<= x0[i];

		af[0] = aX[2];
		af[1] = aX[3];
		af[2] =  (ml*l*pow(aX[3],2)*sin(aX[1]) + ml*g*sin(aX[1])*cos(aX[1]))/(ml*pow(sin(aX[1]),2)+mc);
		af[3] = -(ml*l*pow(aX[3],2)*sin(aX[1])*cos(aX[1]) + (ml + mc)*g*sin(aX[1]))/(l*(ml*pow(sin(aX[1]),2)+mc));

		for (int i = 0; i < n; i++)
			af[i] >>= vf[i];
	}
	trace_off();


	/****************************
	 * Trace for vector field g *
	 ****************************/
	trace_on(TAPE_G);
	{
		for (int i = 0; i < n; i++)
			aX[i] <<= x0[i];

		ag[0] = 0;
		ag[1] = 0;
		ag[2] = 1/(ml*pow(sin(aX[1]),2)+mc);
		ag[3] = -cos(aX[1])/(l*(ml*pow(sin(aX[1]),2)+mc));

		for (int i = 0; i < n; i++)
			ag[i] >>= vg[i];
	}
	trace_off();


	/**********************************
	 * Trace for scalar fields h1, h2 *
	 **********************************/
	trace_on(TAPE_H);
	{
		for (int i = 0; i < n; i++)
			aX[i] <<= x0[i];

		ah[0] = aX[0] + l*sin(aX[1]);
		ah[1] = l*cos(aX[1]);

		for (int i = 0; i < m_H; i++)
			ah[i] >>= vh[i];
	}
	trace_off();


	const int d = 12;

	x0[0] = 1.;
	x0[1] = 0.2;
	x0[2] = -0.5;
	x0[3] = -0.4;

	cout.precision(6); cout << scientific;


	/***************************************************
	 * calculation of Lie derivatives of scalar fields *
	 ***************************************************/

	double** scalar = myalloc2(m_H, d+1);

	cout   << "Lie derivatives:" << endl << endl;

	// calculate Lie derivatives using Lie drivers
	lie_scalar(TAPE_F, TAPE_H, n, m_H, x0, d, scalar);

	for (int i = 0; i <= d; i++)
	{
		for (int j = 0; j < m_H; j++)
			cout   << "Lfh_" << i << "_" << j << " =\t" << scalar[j][i] << endl;
		cout   << endl;
	}

	cout << endl;
	myfree2(scalar);


	/****************************************************************
	 * calculation of gradients of Lie derivatives of scalar fields *
	 ****************************************************************/

	double*** gradient = myalloc3(m_H, n, d+1);

	cout   << "gradients of Lie derivatives:" << endl << endl;

	// calculate gradients of Lie derivatives using Lie drivers
	lie_gradient(TAPE_F, TAPE_H, n, m_H, x0, d, gradient);

	for (int i = 0; i <= d; i++)
	{
		for (int j = 0; j < m_H; j++)
			cout   << "dLfh_" << i << "_" << j << " =\t" << gradient[j][0][i] << "\t" << gradient[j][1][i] << "\t" << gradient[j][2][i] << "\t" << gradient[j][3][i] << endl;
		cout   << endl;
	}

	cout << endl;
	myfree3(gradient);

	
	/*******************************
	 * calculation of Lie brackets *
	 *******************************/

	double** bracket = myalloc2(n, d+1);

	cout   << "Lie brackets:" << endl << endl;

	// calculate Lie brackets using Lie drivers
	lie_bracket(TAPE_F, TAPE_G, n, x0, d, bracket);

	for (int i = 0; i <= d; i++)
	{
		for (int j = 0; j < n; j++)
			cout   << "adfg_" << i << "_" << j << " =\t" << bracket[j][i] << endl;
		cout   << endl;
	}

	myfree2(bracket);
	myfree(x0);

	
	cout << "Press RETURN to continue" << endl;
	cin.get();

	return 0;
}
