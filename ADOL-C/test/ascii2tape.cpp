#include <adolc/adolc.h>
#include <adolc/tapedoc/asciitapes.h>

/* Simple LU-factorization according to Crout's algorithm without pivoting */
void LUfact(int n, adouble **A) {
    int i, j, k;
    adouble dum;
    for (j=0; j<n; j++) { /* L-part */
        for (i=0; i<j; i++)
            for (k=0; k<i; k++)
                A[i][j] -= A[i][k] * A[k][j];
        /* U-part */
        for (i=j; i<n; i++)
            for (k=0; k<j; k++)
                A[i][j] -= A[i][k] * A[k][j];
        if (A[j][j] != 0) {
            dum = 1.0 / A[j][j];
            for (i=j+1; i<n; i++)
                A[i][j] *= dum;
        } else {
            fprintf(stderr,"Error in LUfact(..): pivot is zero\n");
            exit(-99);
        }
    }
}
/* Solution of A*x=b by forward and backward substitution */
void LUsolve(int n, adouble **A, adouble *bx) {
    int i, j;
    /* forward substitution */
    for (i=0; i<n; i++)
        for (j=0; j<i-1; j++)
            bx[i] -= A[i][j] * bx[j];
    /* backward substitution */
    for (i=n-1; i>=0; i--) {
        for (j=i+1; j<n; j++)
            bx[i] -= A[i][j] * bx[j];
        bx[i] /= A[i][i];
    }
}

int main(int argc, char **argv) {
    const int tag   = 1;                       // tape tag
    const int tag2  = 8;
    const int size  = 5;                       // system size
    const int indep = size*size+size;          // # of indeps
    const int depen = size;                    // # of deps

    double  A[size][size], a1[size], a2[size], // passive variables
    b[size], x[size];
    adouble **AA, *AAp, *Abx;                  // active variables
    double *args = myalloc1(indep);            // arguments
    double **jac = myalloc2(depen,indep);      // the Jacobian
    double **jac2 = myalloc2(depen,indep);      // the Jacobian
    double **diff = myalloc2(depen,indep);
    int i,j;


    /*------------------------------------------------------------------------*/
    /* Info */
    fprintf(stdout,"LINEAR SYSTEM SOLVING by "
            "LU-DECOMPOSITION (ADOL-C Example)\n\n");


    /*------------------------------------------------------------------------*/
    /* Taping the computation of the determinant */
    trace_on(tag);
    /*------------------------------------------------------------------------*/
    /* Allocation und initialization of the system matrix */
    AA  = new adouble*[size];
    AAp = new adouble[size*size];
    for (i=0; i<size; i++) {
        AA[i] = AAp;
        AAp += size;
    }
    Abx = new adouble[size];
    for(i=0; i<size; i++) {
        a1[i] = i*0.25;
        a2[i] = i*0.33;
    }
    for(i=0; i<size; i++) {
        for(j=0; j<size; j++)
            A[i][j] = a1[i]*a2[j];
        A[i][i] += i+1;
        b[i] = -i-1;
    }


    /* marking indeps */
    for(i=0; i<size; i++)
        for(j=0; j<size; j++)
            AA[i][j] <<= (args[i*size+j] = A[i][j]);
    for(i=0; i<size; i++)
        Abx[i] <<= (args[size*size+i] = b[i]);
    /* LU-factorization and computation of solution */
    LUfact(size,AA);
    LUsolve(size,AA,Abx);
    /* marking deps */
    for (i=0; i<size; i++)
        Abx[i] >>= x[i];
    trace_off();
    delete[] AA[0];
    delete[] AA;
    delete[] Abx;
    for (i=0; i<size; i++)
        fprintf(stdout," x[%d] (original):  %16.4E\n",i,x[i]);


    /*------------------------------------------------------------------------*/
    /* Recomputation  */
    function(tag,depen,indep,args,x);
    for (i=0; i<size; i++)
        fprintf(stdout," x[%d] (from tape): %16.4E\n",i,x[i]);

    write_ascii_trace("lusolve.txt",tag);
    
    read_ascii_trace("lusolve.txt",tag2);

    /*------------------------------------------------------------------------*/
    /* Recomputation  */
    function(tag2,depen,indep,args,x);
    for (i=0; i<size; i++)
        fprintf(stdout," x[%d] (from tape2): %16.4E\n",i,x[i]);

    /*------------------------------------------------------------------------*/
    /* Computation of Jacobian */
    jacobian(tag,depen,indep,args,jac);
    fprintf(stdout," Jacobian (tape):\n");
    for (i=0; i<depen; i++) {
        for (j=0; j<indep; j++)
            fprintf(stdout," %14.6E",jac[i][j]);
        fprintf(stdout,"\n");
    }

    /*------------------------------------------------------------------------*/
    /* Computation of Jacobian */
    jacobian(tag2,depen,indep,args,jac2);
    fprintf(stdout," Jacobian (tape2):\n");
    for (i=0; i<depen; i++) {
        for (j=0; j<indep; j++) {
            fprintf(stdout," %14.6E",jac2[i][j]);
            diff[i][j] = jac2[i][j] - jac[i][j];
        }
        fprintf(stdout,"\n");
    }
    fprintf(stdout," Jacobian difference:\n");
    for (i=0; i<depen; i++) {
        for (j=0; j<indep; j++) {
            fprintf(stdout," %14.6E",diff[i][j]);
        }
        fprintf(stdout,"\n");
    }
    printTapeStats(stdout,tag);
    printTapeStats(stdout,tag2);
}
