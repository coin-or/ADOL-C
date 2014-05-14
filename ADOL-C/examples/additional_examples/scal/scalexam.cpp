/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     scalexam.cpp
 Revision: $Id$
 Contents:
          This program can be used to verify the consistency and 
          correctness of derivatives computed by ADOL-C in its forward 
          and reverse mode.  
          Ther use is required to select one integer input id. 
          For positive n = id the monomial x^n is evaluated recursively 
          at x=0.5 and all its nonzero Taylor coeffcients at this point 
          are evaluated in the forward and reverse mode. 
          A negative choice of id >= -9 leads to one of nine
          identities, whose derivatives should be trivial. These identities
          may be used to check the correctness of particular code segments
          in the ADOL-C sources uni5_.c and *o_rev.c. No timings are
          performed in this example program.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <math.h>
#include <iostream>
using namespace std;

/****************************************************************************/
/*                                                                    POWER */
/* The monomial evaluation routine which has been obtained from
   the original version by retyping all `doubles' as `adoubles' */
adouble power( adouble x, int n ) {
    adouble z = 1;
    if (n > 0) {
        int nh =n/2;
        z = power(x,nh);
        z *= z;
        if (2*nh != n)
            z *= x;
        return z;
    } else
        if (n == 0)
            return z;
        else
            return 1.0/power(x,-n);
}


/****************************************************************************/
/*                                                                     MAIN */
int main() {
    int n, i, id;
    int tag = 0;
    /*--------------------------------------------------------------------------*/
    fprintf(stdout,"SCALEXAM (ADOL-C Example)\n\n");
    fprintf(stdout,"problem number(-1 .. -10) / degree of monomial =? \n");
    scanf("%d",&id);
    n = id >0 ? id : 3;

    double *xp,*yp;
    xp = new double[n+4];
    yp = new double[n+4];
    yp[0] = 0;
    xp[0] = 0.5;
    xp[1] = 1.0;

    /*--------------------------------------------------------------------------*/
    int dum = 1;
    trace_on(tag,dum);   // Begin taping all calculations with 'adoubles'
    adouble y,x;
    x <<= xp[0];
    if (id >= 0) {
        fprintf(stdout,"Evaluate and differentiate recursive power routine \n");
        y = power(x,n);
    } else {
        fprintf(stdout,
                "Check Operations and Functions by Algebraic Identities \n");
        switch (id) {
            case -1 :
                fprintf(stdout,
                        "Addition/Subtraction: y = x + x - (2.0/3)*x - x/3 \n");
                y =  x + x - (2.0/3)*x - x/3 ;
                break;
            case -2 :
                fprintf(stdout,"Multiplication/divison:  y = x*x/x \n");
                y = x*x/x;
                break;
            case -3 :
                fprintf(stdout,"Square root and power: y = sqrt(pow(x,2)) \n");
                y = sqrt(pow(x,2));
                break;
            case -4 :
                fprintf(stdout,"Exponential and log: y = exp(log(log(exp(x)))) \n");
                y = exp(log(log(exp(x))));
                break;
            case -5 :
                fprintf(stdout,"Trig identity: y = x + sin(2*x)-2*cos(x)*sin(x) \n");
                y =  x + sin(2.0*x)-2.0*cos(x)*sin(x);
                break;
            case -6 :
                fprintf(stdout,"Check out quadrature macro \n");
                y = exp(myquad(myquad(exp(x))));
                break;
            case -7 :
                fprintf(stdout,"Arcsin: y = sin(asin(acos(cos(x)))) \n");
                y = sin(asin(acos(cos(x))));
                break;
            case -8 :
                fprintf(stdout,
                        "Hyperbolic tangent: y = x + tanh(x)-sinh(x)/cosh(x) \n");
                y = x + tanh(x)-sinh(x)/cosh(x) ;
                break;
            case -9 :
                fprintf(stdout,"Absolute value: y = x + fabs(x) - fabs(-x) \n");
                y = x + fabs(-x) - fabs(x);
                break;
            case -10 :
                fprintf(stdout,"atan2: y = atan2(sin(x-0.5+pi),cos(x-0.5+pi)) \n");
                y = atan2(sin(x),cos(x));
                break;
            default :
                fprintf(stdout," Please select problem number >= -10 \n");
                exit(-1);
        }
    }
    y >>= yp[0];
    trace_off();  // The (partial) execution trace is completed.

    /*--------------------------------------------------------------------------*/
    if( id < 0 )
        fprintf(stdout,"Round-off error: %14.6E\n",(y-x).value());

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
    double *res;
    res = new double[n+2];
    double u[1];
    u[0] = 1;
    fprintf(stdout,
            "\nThe two Taylor coefficients in each row should agree\n\n");

    double ***V = (double***)new double**[1];
    V[0] = new double*[1];
    V[0][0] = new double[n+2];
    double **U = new double*[1];
    U[0] = new double[1];
    U[0][0] = 1;
    double** xpoint = &xp;
    double** ypoint = &yp;
    double** respoint = &res;

    // tape_doc(tag,depen,indep,*xpoint,*respoint);

    fprintf(stdout," \n \t   forward  \t    reverse  \n");
    for (i=0; i < n+2; i++) {
        xp[i+2]=0;
        forward(tag,1,1,i,i+1,xpoint,respoint);
        fprintf(stdout,"%d\t%14.6E\t\t%14.6E\n",i,res[i],yp[i]);
        reverse(tag,1,1,i,u,ypoint); // call higher order scalar reverse
        reverse(tag,1,1,i,1,U,V);
        yp[i+1] = yp[i]/(i+1);
        if (V[0][0][i] != yp[i])
            fprintf(stdout,"%d-th component in error %14.6E\n",i,V[0][0][i]-yp[i]);
    }
    cout << "\nWhen n<0 all rows except the first two should vanish \n";

    return 1;
}
