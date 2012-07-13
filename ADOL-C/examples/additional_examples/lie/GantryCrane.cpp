#include "adolc.h"
#include "adolc_lie.h"
#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;

#define TAPE_F 1
#define TAPE_G 2
#define NUM_CALCS 5000

int main()
{
	double x0[4];
	x0[0] = 1.0;
	x0[1] = 2.0;
	x0[2] = 3.0;
	x0[3] = 4.0;

	double vf[4];
	double vg[4];

	adouble aX[4];
	adouble af[4];
	adouble ag[4];

	double mc = 1.0;
	double ml = 1.0;
	double l  = 1.0;
	double g  = 9.81;

	ofstream fout("CompTimeAdolC.txt");
	ofstream resout("Result.txt");

	// Trace for vector field f
	trace_on(TAPE_F);

	aX[0] <<= x0[0];
	aX[1] <<= x0[1];
	aX[2] <<= x0[2];
	aX[3] <<= x0[3];

	af[0] = aX[2];
	af[1] = aX[3];
	af[2] = (ml*l*pow(aX[3],2)*sin(aX[1]) + ml*g*sin(aX[1])*cos(aX[1]))/(ml*pow(sin(aX[1]),2)+mc);
	af[3] = -(ml*l*pow(aX[3],2)*sin(aX[1])*cos(aX[1]) + (ml + mc)*g*sin(aX[1]))/(l*(ml*pow(sin(aX[1]),2)+mc));

	af[0] >>= vf[0];
	af[1] >>= vf[1];
	af[2] >>= vf[2];
	af[3] >>= vf[3];

	trace_off();

	// Trace for vector field g
	trace_on(TAPE_G);

	aX[0] <<= x0[0];
	aX[1] <<= x0[1];
	aX[2] <<= x0[2];
	aX[3] <<= x0[3];

	ag[0] = 0;
	ag[1] = 0;
	ag[2] = 1/(ml*pow(sin(aX[1]),2)+mc);
	ag[3] = -cos(aX[1])/(l*(ml*pow(sin(aX[1]),2)+mc));

	ag[0] >>= vg[0];
	ag[1] >>= vg[1];
	ag[2] >>= vg[2];
	ag[3] >>= vg[3];
    
	trace_off();

	// Calculation of Lie bracket
	int n = 4;
	int d = 12;
	double* pX0 = myalloc(n);
	pX0[0]=1;
	pX0[1]=2;
	pX0[2]=3;
	pX0[3]=4;


	clock_t t0, t1;
	double dt;

	fout << "[";
	for (int i = 0; i <=d; i++)
	{
		double** result = myalloc2(n, i+1);

		t0 = clock();
		for (int k = 0; k < NUM_CALCS; k++)
		{
			lie_bracketv(TAPE_F, TAPE_G, n, pX0, i, result);
		}
		t1 = clock();

		for (int j = 0; j < n; j++)
		{
			cout << "adfg_" << i << "_" << j << " = " << result[j][i] << endl;
			resout << "adfg_" << i << "_" << j << " = " << result[j][i] << endl;
		}

		dt = (double)(t1-t0)/(double(CLOCKS_PER_SEC)*NUM_CALCS);

		cout << "Computation time: " << 1000*dt << " ms" << endl << endl;
		fout << 1000*dt << " ";

		resout << "Computation time: " << 1000*dt << " ms" << endl << endl;

		myfree(result);

	}

	fout << "]";
	myfree(pX0);

	cout << "Press RETURN to continue" << endl;
	cin.get();

	return 0;

}