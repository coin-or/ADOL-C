/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     luexam.cpp
 Revision: $Id$
 Contents: computation of LU factorization with pivoting

 Copyright (c) Kshitij Kulshreshtha
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

#include <adolc/adouble.h>
#include <adolc/advector.h>
#include <adolc/taping.h>

#include <adolc/interfaces.h>
#include <adolc/drivers/drivers.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <sstream>

using namespace std;

adouble findmaxindex(const size_t n, const advector& col, const size_t k) {
    adouble idx = k;
    for (size_t j = k + 1; j < n; j++ )
	condassign(idx,(fabs(col[j]) - fabs(col[idx])), adouble((double)j));
    return idx;
}

// Assuming A is stored row-wise in the vector

void lufactorize(const size_t n, advector& A, advector& p) {
    adouble idx, tmp;
    for (size_t j = 0; j < n; j++)
	p[j] = j;
    for (size_t k = 0; k < n; k++) {
	advector column(n);
	for(size_t j = 0; j < n; j++)
	    condassign(column[j], adouble(double(j - k + 1)), A[j*n + k]);
	idx = findmaxindex(n, column, k);
	tmp = p[k];
	p[k] = p[idx];
	p[idx] = tmp;
	for(size_t j = 0; j < n; j++) {
	    tmp = A[k*n + j];
	    A[k*n + j] = A[idx*n + j];
	    A[idx*n + j] = tmp;
	}
	tmp = 1.0/A[k*n + k];
	for (size_t i = k + 1; i < n; i++) {
	    A[i*n + k] *= tmp;
	    for (size_t j = k + 1; j < n; j++) {
		A[i*n + j] -= A[i*n + k] * A[k*n+j];
	    }
	}
    }
}

void Lsolve(const size_t n, const advector& A, const advector& p, advector& b, advector& x) {
    for (size_t j = 0; j < n; j++) {
	x[j] = b[p[j]];
	for (size_t k = j+1; k <n; k++) {
	    b[p[k]] -= A[k*n+j]*x[j];
	}
    }
}

void Rsolve(const size_t n, const advector& A, advector& x) {
    for (size_t j = 1; j <= n; j++) {
	x[n-j] *=  1.0/A[(n-j)*n + n-j];
	for (size_t k = 0; k < n-j; k++) {
	    x[k] -= A[k*n + n-j]*x[n-j];
	}
    }
}

void printL(const size_t n, const advector& A, ostream &outf = std::cout) {
    for (size_t i = 0; i < n; i++) {
	for (size_t j = 0; j < n; j++)
	    if (j < i)
		outf << setw(8) << A[i*n + j].value() << "  ";
	    else if (j == i)
		outf << setw(8) << 1.0 << "  ";
	    else
		outf << setw(8) << 0.0 << "  ";
	outf << "\n";
    }
}

void printR(const size_t n, const advector& A, ostream &outf = std::cout) {
    for (size_t i = 0; i < n; i++) {
	for (size_t j = 0; j < n; j++)
	    if (j >= i)
		outf << setw(8) << A[i*n + j].value() << "  ";
	    else
		outf << setw(8) << 0.0 << "  ";
	outf << "\n";
    }
}

double norm2(const double *const v, const size_t n)
{
    size_t j;
    double abs,scale,sum;
    scale=0.0;
    sum=0.0;
    for (j=0; j<n; j++) {
	if (v[j] != 0.0) {
	    abs = fabs(v[j]);
	    if (scale <= abs) {
		sum = 1.0 + sum * (scale/abs)*(scale/abs);
		scale = abs;
	    } else
		sum += (abs/scale)*(abs/scale);
	}
    }
    sum = sqrt(sum)*scale;
    return sum;
}

double scalar(double *x, double *y, size_t n)
{
    size_t j;
    int8_t sign;
    double abs,scale,sum,prod;
    scale = 0.0;
    sum = 0.0;
    for(j=0; j<n; j++) {
        sign = 1;
        prod = x[j]*y[j];
        if( prod != 0.0) {
            abs = prod;
            if (abs < 0.0) {
                sign = -1;
                abs = -abs;
            }
            if( scale <= fabs(abs)) {
                sum = sign * 1.0 + sum *(scale/abs);
                scale = abs;
            } else
                sum+=sign*(abs/scale);
        }
    }
    sum = sum*scale;
    return sum;
}

void matvec(const double *const A, const size_t m, const double *const b, const size_t n, double *const ret) 
{
    size_t i,j;
    memset(ret,0,n*sizeof(double));
    for (i=0; i<m; i++) {
	double abs, scale = 0.0, prod, sum = 0.0;
	int8_t sign;
	for (j=0; j<n; j++) {
	    sign = 1;
	    prod = A[i*n+j]*b[j];
	    if (prod != 0.0) {
		abs = prod;
		if (abs < 0.0) {
		    sign = -1;
		    abs = -abs;
		}
		if (scale <=fabs(abs)) {
		    sum = sign * 1.0 + sum * (scale/abs);
		    scale = abs;
		} else
		    sum += sign*(abs/scale);
	    }
	}
	ret[i] = sum*scale;
    }
}

void residue(const double *const A, const size_t m, const double *const b, const size_t n, const double *const x, double *const ret) {
    double *b2 = new double[n];
    matvec(A,m,x,n,b2);
    for (size_t i = 0; i < n; i++)
	ret[i] = b[i] - b2[i];
    delete[] b2;
}

double normresidue(const double *const A, const size_t m, const double *const b, const size_t n, const double*const x) {
    double *res = new double[n];
    residue(A,m,b,n,x,res);
    double ans = norm2(res, n);
    delete[] res;
    return ans;
}

int main() {
    int tag = 1;
    int keep = 1;
    int n;
    string matrixname, rhsname;
    ifstream matf, rhsf;
    double *mat, *rhs, *ans, err, normx, normb;

    cout << "COMPUTATION OF LU-Factorization with pivoting (ADOL-C Documented Example)\n\n";
    cout << "order of matrix = ? \n"; // select matrix size
    cin >> n;

    cout << "---------------------------------\nNow tracing:\n";
    rhs = new double[n*n + n]; 
    mat = rhs + n;
    ans = new double[n];
    cout << "file name for matrix = ?\n";
    cin >> matrixname;
    cout << "file name for rhs = ?\n";
    cin >> rhsname;


    matf.open(matrixname.c_str());
    for (size_t i = 0; i < n*n; i++)
	matf >> mat[i];
    matf.close();

    rhsf.open(rhsname.c_str());
    for (size_t i = 0; i < n; i++)
	rhsf >> rhs[i];
    rhsf.close();

    {
	trace_on(tag,keep);               // tag=1=keep
	advector A(n*n), b(n), x(n), p(n);
	for(size_t i = 0; i < n; i++)
	    b[i] <<= rhs[i];
	for(size_t i = 0; i < n*n; i++)
	    A[i] <<= mat[i];
	lufactorize(n, A, p);
	Lsolve(n, A, p, b, x);
	Rsolve(n, A, x);
	for(size_t i = 0; i < n; i++)
	    x[i] >>= ans[i];
	trace_off();
    }
    
    err = normresidue(mat, n, rhs, n, ans);
    normb = norm2(rhs, n);
    normx = norm2(ans, n);
    cout << "Norm rhs = " << normb <<"\n"; 
    cout << "Norm solution = " << normx <<"\n"; 
    cout << "Norm of residue = " << err <<"\t relative error = " << err/normx << "\n";

    cout << "---------------------------------\nNow computing from trace:\n";
    cout << "file name for matrix = ?\n";
    cin >> matrixname;
    cout << "file name for rhs = ?\n";
    cin >> rhsname;


    matf.open(matrixname.c_str());
    for (size_t i = 0; i < n*n; i++)
	matf >> mat[i];
    matf.close();

    rhsf.open(rhsname.c_str());
    for (size_t i = 0; i < n; i++)
	rhsf >> rhs[i];
    rhsf.close();

    zos_forward(tag, n, n*n + n, keep, rhs, ans);

    err = normresidue(mat, n, rhs, n, ans);
    normb = norm2(rhs, n);
    normx = norm2(ans, n);
    cout << "Norm rhs = " << normb <<"\n"; 
    cout << "Norm solution = " << normx <<"\n"; 
    cout << "Norm of residue = " << err <<"\t relative error = " << err/normx <<"\n";
    double *ansbar = new double[n];
    double *matcol = new double[n];
    double *rhsbar = new double[n*n+n];
    double *matbar = rhsbar + n;
    double scprod = 0.0;

    memset(rhsbar, 0, (n*n+n)*sizeof(double));
    memset(ansbar, 0, n*sizeof(double));
    for (size_t k = 0; k < n; k++) {
    cout << "computing gradient of element " << k + 1 << " of solution w.r.t. matrix elements and rhs\n";
    ansbar[k] = 1.0;

    fos_reverse(tag, n, n*n+n, ansbar, rhsbar);

    for (size_t i = 0; i < n*n + n; i++)
	cout << "bar[" << i << "] = " <<  rhsbar[i] << "\n";

    for (size_t j = 0; j < n; j++) {
	for (size_t i = 0; i < n; i++)
	    matcol[i] = mat[i*n + j];
	scprod = scalar(rhsbar, matcol, n);
	cout << "gradient w.r.t. rhs times column " << j + 1 << " of matrix  = " << scprod << "\n";
    }
    ansbar[k] = 0.0;
    }
    delete[] ansbar;
    delete[] matcol;
    delete[] rhsbar;

    delete[] rhs;
    delete[] ans;
}
