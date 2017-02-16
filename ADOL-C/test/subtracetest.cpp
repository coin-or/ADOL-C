#include <adolc/tapedoc/asciitapes.h>
#include <adolc/interfaces.h>

int main() {
	int n = 2, m = 2;
	short tag = 1, finaltag;
	finaltag = read_ascii_trace("first_adolcAsciiTrace.txt",tag);
	double *x = new double[n];
	double *y = new double[m];
	for (int i = 0; i < n ; i++ ) x[i] = 1.0;
	zos_forward(tag,m,n,0,x,y);
	for (int i = 0; i < m ; i++ ) printf("y[%d] = %lf\n",i,y[i]);
	printf("final tag = %d\n", finaltag);
	return 0;
}
	
