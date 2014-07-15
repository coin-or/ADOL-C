/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     vgenodemain.cpp
 Revision: $Id$
 Contents: example for ODE Generic Main File: 
       for use with function modules containing vector examples 
       (e.g. vgen_eutroph.C)
 
   Each << function module >> contains:
          
     (1) const char* const controlFileName 
     (2) int indepDim; 
     (3) int depDim; 
     (4) void initProblemParameters( void )
     (5) void initIndependents( double* indEPS_ )
     (6) void originalVectorFunction( double* indEPS_, double* dEPS_ )
     (7) void tapingVectorFunction( int tag, double* indEPS_, double* dEPS_ )   

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

#include <math.h>


/****************************************************************************/
/*                                                                   MACROS */
#define TIMEFORMAT " %12.6E units,   %12.6E scunits,   %12.6E seconds\n"


/****************************************************************************/
/*                                      EXTERNAL STUFF FROM FUNCTION MODULES*/

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
extern int indepDim;
extern int depDim;

/*--------------------------------------------------------------------------*/
/*                                                  Init Problem Parameters */
extern void initProblemParameters( void );

/*--------------------------------------------------------------------------*/
/*                                                        Initialize indeps */
extern void initIndependents( double* indeps );

/*--------------------------------------------------------------------------*/
/*                                                             original RHS */
extern void originalVectorFunction( double* indeps, double* deps );

/*--------------------------------------------------------------------------*/
/*                                                               taping RHS */
extern void tapingVectorFunction( int tag, double* indeps, double* deps );


/****************************************************************************/
/*                                                                     MAIN */
int main() {
    int degree, evalCount, taskCount;
    int tag = 1;                  /* tape tag */
    int i, j, k, yes;
    double tau;

    fprintf(stdout,"GENERIC ODE EXAMPLE (ADOL-C Example)\n\n");
    /*------------------------------------------------------------------------*/
    /* Initialize all problem parameters (including  dimension) */
    initProblemParameters();
    if (indepDim != depDim) {
        fprintf(stdout," ERROR indepDim (%d) != depDim (%d)\n",indepDim,depDim);
        exit(-1);
    }

    /*------------------------------------------------------------------------*/
    /* Check main parameters */
    fprintf(stdout," Highest derivatives =? \n ");
    scanf("%d",&degree);
    fprintf(stdout," Number of evaluations =? \n ");
    scanf("%d",&evalCount);
    fprintf(stdout," Nonzero scaling paramater =?\n ");
    scanf("%le",&tau);

    /*------------------------------------------------------------------------*/
    /* Initialize the independent variables */
    double*  indeps  = new double[indepDim];
    double*  deps    = new double[depDim];
    double** indeps2 = myalloc(indepDim,degree+1);
    initIndependents(indeps);
    for (i=0; i<indepDim; i++) {
        indeps2[i][0] = indeps[i];
        for (j=1; j<=degree; j++)
            indeps2[i][j] = 0.0;
    }

    /*------------------------------------------------------------------------*/
    /* Necessary variable */
    double*** B = myalloc(indepDim,indepDim,degree);
    double*** A = myalloc(indepDim,indepDim,degree);
    double **w  = myalloc(indepDim,degree+1);
    ;
    short **nonzero = new short*[indepDim];
    for (i=0; i<indepDim; i++)
        nonzero[i] = new short[indepDim];


    /****************************************************************************/
    /*                                          0. ORIGINAL FUNCTION EVALUATION */
    /*                                             ---> always                  */
    taskCount = 0;
    fprintf(stdout,"\nTASK %d: Original function evaluation\n",
            taskCount++);

    double t00 = myclock();
    for (i=1; i<=evalCount; i++) {
        indeps[1] = 1.0/i;
        originalVectorFunction(indeps,deps);
    }
    double t01 = myclock();

    double rtu, stu;
    if (t01-t00) {
        stu = 2.0/(t01-t00)/((double)degree)/(((double)degree)+1.0);
        rtu = 1.0/(t01-t00);
        fprintf(stdout,"          ");
        fprintf(stdout,TIMEFORMAT,1.0,2.0/((double)degree)/(((double)degree)+1.0),
                (t01-t00)/evalCount);
    } else {
        fprintf(stdout,"    !!! zero timing !!!\n");
        fprintf(stdout,"    set time unit to 1.0\n");
        rtu = 1.0;
        stu = 2.0/((double)degree)/(((double)degree)+1.0);
    }


    /****************************************************************************/
    /*                                                   1. TAPING THE FUNCTION */
    /*                                                      ---> always         */
    fprintf(stdout,"--------------------------------------------------------");
    fprintf(stdout,"\nTASK %d: Taping the function\n",
            taskCount++);

    t00 = myclock(1);
    /* NOTE: taping will be performed ONCE only */
    tapingVectorFunction(tag,indeps,deps);
    t01 = myclock();

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

    fprintf(stdout,"           ");
    fprintf(stdout,TIMEFORMAT,(t01-t00)*rtu*evalCount,
            (t01-t00)*stu*evalCount,
            (t01-t00));

    /****************************************************************************/
    /*                                                                2. FORODE */
    fprintf(stdout,"--------------------------------------------------------");
    fprintf(stdout,"\nTASK %d: forode(tag, n=%d, tau=%f, dnew=%d, X[n][d+1])\n",
            taskCount++,indepDim,tau,degree);
    t00 = myclock();
    for (j=0; j<evalCount; j++)
        forode(tag,indepDim,tau,degree,indeps2);
    t01 = myclock();

    fprintf(stdout,TIMEFORMAT,(t01-t00)*rtu,
            (t01-t00)*stu,
            (t01-t00)/evalCount);

    /****************************************************************************/
    /*                                                               3. REVERSE */
    fprintf(stdout,"--------------------------------------------------------");
    fprintf(stdout,"\nTASK %d: reverse(tag, m=%d, n=%d, d=%d, Z[n][n][d+1],"
            " nz[n][n])\n",
            taskCount++,depDim,indepDim,degree-1);

    t00 = myclock();
    for (j=0; j<evalCount; j++)
        reverse(tag,depDim,indepDim,degree-1,A,nonzero);
    t01 = myclock();

    fprintf(stdout,TIMEFORMAT,(t01-t00)*rtu,
            (t01-t00)*stu,
            (t01-t00)/evalCount);

    /****************************************************************************/
    /*                                                                 4.ACCODE */
    fprintf(stdout,"--------------------------------------------------------");
    fprintf(stdout,"\nTASK %d: accode(tag, n=%d, tau=%f, d=%d, Z[n][n][d+1],"
            " B[n][n][d+1], nz[n][n])\n",
            taskCount++,indepDim,tau,degree-1);

    t00 = myclock();
    for (j=0; j<evalCount; j++)
        accode(indepDim,tau,degree-1,A,B,nonzero);
    t01 = myclock();

    fprintf(stdout,TIMEFORMAT,(t01-t00)*rtu,
            (t01-t00)*stu,
            (t01-t00)/evalCount);


    /****************************************************************************/
    /*                                                               5. RESULTS */
    fprintf(stdout,"--------------------------------------------------------");
    fprintf(stdout,"\nTASK %d: CHECK results\n\n",taskCount++);

    /*------------------------------------------------------------------------*/
    fprintf(stdout," Print out the nonzero pattern? (0/1)\n ");
    scanf("%d",&yes);
    if (yes) {
        fprintf(stdout," 4 = transcend , 3 = rational , 2 = polynomial ,"
                " 1 = linear , 0 = zero \n");
        fprintf(stdout," negative number k indicate that entries of all "
                "B_j with j < -k vanish  \n");
        for (i=0; i<indepDim; i++) {
            for (j=0; j<indepDim; j++)
                fprintf(stdout,"%3d ",nonzero[i][j]);
            fprintf(stdout,"\n");
        }
    }

    /*------------------------------------------------------------------------*/
    /* The D+1 columns of z should now be consistent with the
       ODE as represented by the time. Feeding z into forward
       we obtain a coeffient array w, whose columns should
       equal to the shifted and scaled columns of z */
    fprintf(stdout,"\n Check that forward reproduces the Taylor series \n");
    forward(tag,depDim,indepDim,degree-1,degree,indeps2,w);
    double err = 0, avg;
    for (i=0; i<degree; i++)
        for (j=0; j<indepDim; j++) {
            avg = (fabs(w[j][i]*tau)+fabs(indeps2[j][i+1]*(i+1)))/2.0;
            if (avg < 0.1)
                avg = 0.1;
            err += fabs(w[j][i]*tau-indeps2[j][i+1]*(i+1))/avg;
        }
    fprintf(stdout,"%14.6E = total error \n",err);

    /*------------------------------------------------------------------------*/
    /* If desired print out Jacobians of Taylor coeffcients with
       respect to the base point */
    fprintf(stdout,"\n Print Jacobian of Taylor coefficient vectors? (0/1)\n ");
    scanf("%d",&yes);
    if (yes) {
        for (i=0; i<degree; i++) {
            fprintf(stdout,"\n\t<-- B(%d)\n",i);
            for (j=0; j<indepDim; j++) {
                for (k=0; k<indepDim; k++)
                    fprintf(stdout,"%14.6E ",B[j][k][i]);
                fprintf(stdout,"\n");
            }
        }
    }

    /*------------------------------------------------------------------------*/
    fprintf(stdout,"\n Increment for differencing, skipped if zero =?\n ");
    double h;
    scanf("%le",&h);
    for (i=0; i<indepDim; i++)
        *w[i] = *indeps2[i];
    if (h != 0)
        for (i=0; i<degree; i++) {
            err = 0;
            for (k=0; k<indepDim; k++) {
                *w[k] += h;
                forode(tag,indepDim,tau,degree,w);
                *w[k] -= h;
                for (j=0; j<indepDim; j++)
                    err += B[j][k][i] != 0 ?
                           fabs(1-(w[j][i+1]-indeps2[j][i+1])/h/B[j][k][i])
                           : fabs((w[j][i+1]-indeps2[j][i+1])/h) ;
            }
            fprintf(stdout," Relative truncation errors in B(%d) ---> %14.6E\n",
                    i,err);
        }

    return 1;
}


/****************************************************************************/
/*                                                               THAT'S ALL */

