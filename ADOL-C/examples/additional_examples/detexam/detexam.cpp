/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     detexam.cpp
 Revision: $Id$
 Contents: modified computation of determinants

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>
#include "../clock/myclock.h"


/****************************************************************************/
/*                                                           DOUBLE ROUTINE */
int n,it;
double** PA;
double pdet( int k, int m ) {
    if (m == 0)
        return 1.0 ;
    else {
        double* pt = PA[k-1];
        double t = 0;
        int p = 1;
        int s;
        if (k%2)
            s = 1;
        else
            s = -1;
        for (int i=0; i<n; i++) {
            int p1 = 2*p;
            if (m%p1 >= p) {
                if (m == p) {
                    if (s>0)
                        t += *pt;
                    else
                        t -= *pt;
                } else {
                    if (s>0)
                        t += *pt*pdet(k-1, m-p);
                    else
                        t -= *pt*pdet(k-1, m-p);
                }
                s = -s;
            }
            ++pt;
            p = p1;
        }
        return t;
    }
}

/****************************************************************************/
/*                                                          ADOUBLE ROUTINE */
adouble** A;
adouble zero = 0;
adouble det( int k, int m ) {
    if (m == 0)
        return 1.0;
    else {
        adouble* pt = A[k-1];
        adouble t = zero;
        int p = 1;
        int s;
        if (k%2)
            s = 1;
        else
            s = -1;
        for (int i=0; i<n; i++) {
            int p1 = 2*p;
            if (m%p1 >= p) {
                if (m == p) {
                    if (s>0)
                        t += *pt;
                    else
                        t -= *pt;
                } else {
                    if (s>0)
                        t += *pt*det(k-1, m-p);
                    else
                        t -= *pt*det(k-1, m-p);
                }
                s = -s;
            }
            ++pt;
            p = p1;
        }
        return t;
    }
}

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
    int i, j;
    int tag = 1;
    fprintf(stdout,"COMPUTATION OF DETERMINANTS Type 1 (ADOL-C Example)\n\n");
    fprintf(stdout,"order of matrix = ? \n");
    scanf("%d",&n);
    A  = new adouble*[n];
    PA = new double*[n];
    int n2 = n*n;
    double* a = new double[n2];

    /*--------------------------------------------------------------------------*/
    /* Preparation */
    double diag = 0;
    int m = 1;
    double* pa = a;
    for (i=0; i<n; i++) {
        m *= 2;
        PA[i] = new double[n];
        double* ppt = PA[i];
        for (j=0; j<n; j++) {
            *ppt++ = j/(1.0+i);
            *pa++  = j/(1.0+i);
        }
        diag += PA[i][i];   // val corrected to value 2/23/91
        PA[i][i] += 1.0;
        a[i*n+i] += 1.0;
    }
    diag += 1;

    /*--------------------------------------------------------------------------*/
    double t00 = myclock();                               /* 0. time (taping) */
    trace_on(tag);
    for (i=0; i<n; i++) {
        A[i] = new adouble[n];
        adouble* pt = A[i];
        double* ppt = PA[i];
        for (j=0; j<n; j++)
            *pt++ <<= *ppt++;
    }
    adouble deter;
    deter = det(n,m-1);
    double detout = 0.0;
    deter >>= detout;
    trace_off();
    double t01 = myclock();
    fprintf(stdout,"\n %f =? %f should be the same \n",detout,diag);

    /*--------------------------------------------------------------------------*/
    size_t tape_stats[STAT_SIZE];
    tapestats(tag,tape_stats);

    fprintf(stdout,"\n    independents            %zu\n",tape_stats[NUM_INDEPENDENTS]);
    fprintf(stdout,"    dependents              %zu\n",tape_stats[NUM_DEPENDENTS]);
    fprintf(stdout,"    operations              %zu\n",tape_stats[NUM_OPERATIONS]);
    fprintf(stdout,"    operations buffer size  %zu\n",tape_stats[OP_BUFFER_SIZE]);
    fprintf(stdout,"    locations buffer size   %zu\n",tape_stats[LOC_BUFFER_SIZE]);
    fprintf(stdout,"    constants buffer size   %zu\n",tape_stats[VAL_BUFFER_SIZE]);
    fprintf(stdout,"    maxlive                 %zu\n",tape_stats[NUM_MAX_LIVES]);
    fprintf(stdout,"    valstack size           %zu\n\n",tape_stats[TAY_STACK_SIZE]);

    /*--------------------------------------------------------------------------*/
    int itu = 8-n;
    itu = itu*itu*itu*itu;
    itu = itu > 0 ? itu : 1;
    double raus;

    /*--------------------------------------------------------------------------*/
    double t10 = myclock();                             /* 1. time (original) */
    for (it = 0; it < itu; it++)
        raus = pdet(n,m-1);
    double t11 = myclock();
    double rtu = itu/(t11-t10);

    double* B = new double[n2];
    double* detaut = new double[1];

    /*--------------------------------------------------------------------------*/
    double t40 = myclock();                      /* 4. time (forward no keep) */
    for (it = 0; it < itu; it++)
        forward(tag,1,n2,0,a,detaut);
    double t41 = myclock();

    /*--------------------------------------------------------------------------*/
    double t20 = myclock();                         /* 2. time (forward+keep) */
    for(it = 0; it < itu; it++)
        forward(tag,1,n2,1,a,detaut);
    double t21 = myclock();
    // fprintf(stdout,"\n %f =? %f should be the same \n",detout,*detaut);

    double u[1];
    u[0] = 1.0;

    /*--------------------------------------------------------------------------*/
    double t30 = myclock();                              /* 3. time (reverse) */
    for (it = 0; it < itu; it++)
        reverse(tag,1,n2,0,u,B);
    double t31 = myclock();

    /*--------------------------------------------------------------------------*/
    /* output of results */
    // optional generation of tape_doc.tex
    // tape_doc(tag,1,n2,a,detaut);
    fprintf(stdout,"\n first base? :   \n");
    for (i=0; i<n; i++) {
        adouble sum = 0;
        adouble* pt;
        pt = A[i];
        for (j=0; j<n; j++)
            sum += (*pt++)*B[j];
        fprintf(stdout,"%E ",sum.value());
    }
    fprintf(stdout,"\n\n times for ");
    fprintf(stdout,"\n tracing          : \t%E",(t01-t00)*rtu);
    fprintf(stdout," units \t%E    seconds",(t01-t00));
    fprintf(stdout,"\n forward (no keep): \t%E",(t41-t40)*rtu/itu);
    fprintf(stdout," units \t%E    seconds",(t41-t40)/itu);
    fprintf(stdout,"\n forward + keep   : \t%E",(t21-t20)*rtu/itu);
    fprintf(stdout," units \t%E    seconds",(t21-t20)/itu);
    fprintf(stdout,"\n reverse          : \t%E",(t31-t30)*rtu/itu);
    fprintf(stdout," units \t%E    seconds\n",(t31-t30)/itu);

    return 1;
}
