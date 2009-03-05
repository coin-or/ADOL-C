/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     speelpenning.cpp
 Revision: $Id: speelpenning.cpp 300 2009-02-26 10:42:44Z awalther $
 Contents: example for computation of sparse hessians

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
   
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/


/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>
#include <adolc/../examples/additional_examples/clock/myclock.h>


/****************************************************************************/
/*                                                                   MACROS */
#define abs(x) ((x >= 0) ? (x) : -(x))
#define maxabs(x,y) (((x)>abs(y)) ? (x) : abs(y))
#define TAG 1


/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
    int n,i,it;
    int tape_stats[STAT_SIZE];

    /*--------------------------------------------------------------------------*/
    /* Input */
    fprintf(stdout,"SPEELPENNINGS PRODUCT Type 1 (ADOL-C Example)\n\n");
    fprintf(stdout,"number of independent variables = ?  \n");
    scanf("%d",&n);
    int itu;
    fprintf(stdout,"number of evaluations = ?  \n");
    scanf("%d",&itu);

    /*--------------------------------------------------------------------------*/
    double yp=0.0;                 /* 0. time  (undifferentiated double code) */
    double *xp = new double[n];
    /* Init */
    for (i=0;i<n;i++)
        xp[i] = (i+1.0)/(2.0+i);

    double t00 = myclock(1);
    for (it=0; it<itu; it++) {
        yp = 1.0;
        for (i=0; i<n; i++)
            yp *= xp[i];
    }
    double t01 = myclock();

    /*--------------------------------------------------------------------------*/
    double yout=0;                             /* 1. time (tracing ! no keep) */

    double t10 = myclock();
    trace_on(TAG);
    adouble* x;
    x = new adouble[n];
    adouble y;
    y = 1;
    for (i=0; i<n; i++) {
        x[i] <<= xp[i];
        y *= x[i];
    }
    y >>= yout;
    delete [] x;
    trace_off();
    double t11 = myclock();

    fprintf(stdout,"%le =? %le  function values should be the same \n",yout,yp);

    /*--------------------------------------------------------------------------*/
    tapestats(TAG,tape_stats);

    fprintf(stdout,"\n    independents            %d\n",tape_stats[NUM_INDEPENDENTS]);
    fprintf(stdout,"    dependents              %d\n",tape_stats[NUM_DEPENDENTS]);
    fprintf(stdout,"    operations              %d\n",tape_stats[NUM_OPERATIONS]);
    fprintf(stdout,"    operations buffer size  %d\n",tape_stats[OP_BUFFER_SIZE]);
    fprintf(stdout,"    locations buffer size   %d\n",tape_stats[LOC_BUFFER_SIZE]);
    fprintf(stdout,"    constants buffer size   %d\n",tape_stats[VAL_BUFFER_SIZE]);
    fprintf(stdout,"    maxlive                 %d\n",tape_stats[NUM_MAX_LIVES]);
    fprintf(stdout,"    valstack size           %d\n\n",tape_stats[TAY_STACK_SIZE]);

    /*--------------------------------------------------------------------------*/
    double **r = new double*[1];
    r[0] = new double[1];
    r[0][0] = yp;
    double err;
    double *z = new double[n];
    double *g = new double[n];
    double* h = new double[n];
    double *ind = new double[n];

    /*--------------------------------------------------------------------------*/
    double t60 = myclock();                      /* 6. time (forward no keep) */
    for (it=0; it<itu; it++)
        forward(TAG,1,n,0,xp,*r);
    double t61 = myclock();

    /*--------------------------------------------------------------------------*/
    double t20 = myclock();                         /* 2. time (forward+keep) */
    for (it=0; it<itu; it++)
        forward(TAG,1,n,1,xp,*r);
    double t21 = myclock();

    /*--------------------------------------------------------------------------*/
    double t30 = myclock();                              /* 3. time (reverse) */
    for (it=0; it<itu; it++)
        reverse(TAG,1,n,0,1.0,g);
    double t31 = myclock();

    err=0;
    for (i=0; i<n; i++) // Compare with deleted product
    { err = maxabs(err,xp[i]*g[i]/r[0][0] - 1.0);
        ind[i] = xp[i];
    }

    fprintf(stdout,"%le = maximum relative errors in gradient (fw+rv)\n",err);

    /*--------------------------------------------------------------------------*/
    double t40 = myclock();                             /* 4. time (gradient) */
    for (it=0; it<itu; it++)
        gradient(TAG,n,ind,z);  //last argument lagrange is ommitted
    double t41 = myclock();

    err = 0;
    for (i=0; i<n; i++)  // Compare with previous numerical result
        err =  maxabs(err,g[i]/z[i] - 1.0);

    fprintf(stdout,"%le = gradient error should be exactly zero \n",err);

    /*--------------------------------------------------------------------------*/
    double *tan = new double[n];            /* 5. time (first row of Hessian) */
    for (i=1; i<n; i++)
        tan[i] = 0.0 ;
    tan[0]=1.0;

    double t50 = myclock();
    for (it=0; it<itu; it++)
        hess_vec(TAG,n,ind,tan,h);  // Computes Hessian times direction tan.
    double t51 = myclock();

    err = abs(h[0]);
    for (i=1; i<n; i++) //Compare with doubly deleted product
        err = maxabs(err,xp[0]*h[i]/g[i]-1.0);

    fprintf(stdout,"%le = maximum relative error in Hessian column \n",err);

    /*--------------------------------------------------------------------------*/
    double h1n = h[n-1];                                /* Check for symmetry */
    tan[0]=0;
    tan[n-1]=1;
    hess_vec(TAG,n,ind,tan,h);   // Computes Hessian times direction tan.

    fprintf(stdout,
            "%le = %le (1,n) and (n,1) entry should be the same\n",h1n,h[0]);

    /*--------------------------------------------------------------------------*/
    /* output of results */
    if (t01-t00) {
        double rtu = 1.0/(t01-t00);
        fprintf(stdout,"\n\n times for ");
        fprintf(stdout,"\n unitime          : \t%le  seconds",(t01-t00)/itu);
        fprintf(stdout,"\n tracing          : \t%le",(t11-t10)*rtu*itu);
        fprintf(stdout,"   units \t%le seconds",(t11-t10));
        fprintf(stdout,
                "\n----------------------------------------------------------");
        fprintf(stdout,"\n forward (no keep): \t%le",(t61-t60)*rtu);
        fprintf(stdout,"   units \t%le seconds",(t61-t60)/itu);
        fprintf(stdout,"\n forward + keep   : \t%le",(t21-t20)*rtu);
        fprintf(stdout,"   units \t%le seconds",(t21-t20)/itu);
        fprintf(stdout,"\n reverse          : \t%le",(t31-t30)*rtu);
        fprintf(stdout,"   units \t%le seconds",(t31-t30)/itu);
        fprintf(stdout,
                "\n----------------------------------------------------------");
        fprintf(stdout,"\n gradient         : \t%le",(t41-t40)*rtu);
        fprintf(stdout,"   units \t%le seconds",(t41-t40)/itu);
        fprintf(stdout,"\n hess*vec         : \t%le",(t51-t50)*rtu);
        fprintf(stdout,"   units \t%le seconds\n",(t51-t50)/itu);
    } else
        fprintf(stdout,"\n-> zero timing due to small problem dimension \n");

    return 1;
}

