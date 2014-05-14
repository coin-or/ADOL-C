/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sgenmain.cpp
 Revision: $Id$
 Contents: Scalar Generic Main File: 
       for use with function modules containing several scalar
       examples 
       (e.g. the determinant example in sfunc_determinant.cpp)

   Each << function module >> contains:
          
     (1) const char* const controlFileName 
     (2) int indepDim; 
     (3) void initProblemParameters( void )
     (4) void initIndependents( double* indeps )
     (5) double originalScalarFunction( double* indeps )
     (6) double tapingScalarFunction( int tag, double* indeps )   
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#define _SGENMAIN_C_

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>
#include "../clock/myclock.h"

#include <cstdlib>
#include <time.h>


/****************************************************************************/
/*                                                                   MACROS */
#define TIMEFORMAT " %12.6E units,   %12.6E seconds\n"


/****************************************************************************/
/*                                      EXTERNAL STUFF FROM FUNCTION MODULES*/

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const extern char* controlFileName;

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
extern int indepDim;

/*--------------------------------------------------------------------------*/
/*                                                  Init Problem Parameters */
extern void initProblemParameters( void );

/*--------------------------------------------------------------------------*/
/*                                                        Initialize indeps */
extern void initIndependents( double* indeps );

/*--------------------------------------------------------------------------*/
/*                                                 Original scalar function */
extern double originalScalarFunction( double* indeps );

/*--------------------------------------------------------------------------*/
/*                                                   Taping scalar function */
extern double tapingScalarFunction( int tag, double* indeps );


/****************************************************************************/
/*                                                            CONTROL STUFF */
enum controlParameter {
    cpDimension,
    cpAverageCount,
    cpDegree,
    cpVecCountFW,
    cpVecCountRV,
    cpVecCountTR,
    cpZosFW,
    cpFosFW,
    cpHosFW,
    cpFovFW,
    cpHovFW,
    cpFosRV,
    cpHosRV,
    cpFovRV,
    cpHovRV,
    cpFunction,
    cpJacobian,
    cpVecJac,
    cpJacVec,
    cpHessian,
    cpHessVec,
    cpLagHessVec,
    cpTensor,
    cpInvTensor,
    cpCount
};


/****************************************************************************/
/*                                                     PROVIDE RANDOM INITs */
//unsigned short int dx[3]; /* variable needed by erand48(.) */

void initRand ( void )  /* a function to initialize dx using actual time */
{ struct tm s;
    time_t t;
    time(&t);
    s=*localtime(&t);
    srand(s.tm_sec*s.tm_min);
    /*  dx[0]=rand();
      dx[1]=rand();
      dx[2]=rand();*/
}


/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
    int i, j, k;
    int tag = 1;                  /* tape tag */
    int taskCount = 0;

    int pFW, pRV, pTR, degree, keep;   /* forward/reverse parameters */
    int evalCount;                     /* # of evaluations */


    /****************************************************************************/
    /*                                        READ CONTROL PARAMETERS FROM FILE */
    int controlParameters[cpCount];
    FILE* controlFile;

    /*------------------------------------------------------------------------*/
    /*                                                      open file to read */
    if ((controlFile = fopen(controlFileName,"r")) == NULL) {
        fprintf(stdout,"ERROR: Could not open control file %s\n",
                controlFileName);
        exit(-1);
    }

    /*------------------------------------------------------------------------*/
    /*                                                        read all values */
    for (i=0; i<cpCount; i++)
        fscanf(controlFile,"%d%*[^\n]",&controlParameters[i]);

    indepDim  = controlParameters[cpDimension];
    pFW       = controlParameters[cpVecCountFW];
    pRV       = controlParameters[cpVecCountRV];
    pTR       = controlParameters[cpVecCountTR];
    degree    = controlParameters[cpDegree];
    evalCount = controlParameters[cpAverageCount];

    /*------------------------------------------------------------------------*/
    /*                                                     close control file */
    fclose(controlFile);


    /****************************************************************************/
    /*                                               VARIABLES & INITIALIZATION */

    /*------------------------------------------------------------------------*/
    /* Initialize all problem parameters (including  dimension) */
    initProblemParameters();

    /*------------------------------------------------------------------------*/
    /* Initialize the independent variables */
    double* indeps = new double[indepDim];
    initIndependents(indeps);

    /*------------------------------------------------------------------------*/
    /* Check main parameters */
    if (evalCount <= 0) {
        fprintf(stdout,"    # of evaluations to average over = ? ");
        fscanf(stdin,"%d",&evalCount);
        fprintf(stdout,"\n");
    }

    if ((degree <= 1) &&
            (controlParameters[cpHosFW] || controlParameters[cpHovFW] ||
             controlParameters[cpHosRV] || controlParameters[cpHovRV] ||
             controlParameters[cpTensor])) {
        fprintf(stdout,"    degree = ? ");
        fscanf(stdin,"%d",&degree);
        fprintf(stdout,"\n");
    }
    keep = degree + 1;

    if ((pFW < 1) &&
            (controlParameters[cpFovFW] || controlParameters[cpHovFW])) {
        fprintf(stdout,"    # of vectors in vector forward mode = ? ");
        fscanf(stdin,"%d",&pFW);
        fprintf(stdout,"\n");
    }

    if ((pRV < 1) &&
            (controlParameters[cpFovRV] || controlParameters[cpHovRV])) {
        fprintf(stdout,"    # of vectors in vector reverse mode = ? ");
        fscanf(stdin,"%d",&pRV);
        fprintf(stdout,"\n");
    }

    if ((pTR < 1) &&
            (controlParameters[cpTensor])) {
        fprintf(stdout,"    # of vectors in tensor mode = ? ");
        fscanf(stdin,"%d",&pTR);
        fprintf(stdout,"\n");
    }

    /*------------------------------------------------------------------------*/
    /* Necessary variable */
    double depOrig=0.0, depTape;    /* function value */
    double ***XPPP, **XPP;
    double ***YPPP, **YPP, *YP;
    double ***ZPPP, **ZPP, *ZP;
    double          *UP, u;
    double                 *VP;
    double                 *WP;
    double          *JP;
    short           **nzPP;
    int retVal=0;                 /* return value */
    double t00, t01, t02, t03;  /* time values */
    double          **TPP;
    double          **SPP;
    double          **HPP;
    int dim;


    /****************************************************************************/
    /*                                                          NORMALIZE TIMER */



    /****************************************************************************/
    /*                                          0. ORIGINAL FUNCTION EVALUATION */
    /*                                             ---> always                  */
    fprintf(stdout,"\nTASK %d: Original function evaluation\n",
            taskCount++);

    t00 = myclock();
    for (i=0; i<evalCount; i++)
        depOrig = originalScalarFunction(indeps);
    t01 = myclock();

    double timeUnit;
    if (t01-t00) {
        timeUnit = 1.0/(t01-t00);
        fprintf(stdout,"          ");
        fprintf(stdout,TIMEFORMAT,1.0,
                (t01-t00)/evalCount);
    } else {
        fprintf(stdout,"    !!! zero timing !!!\n");
        fprintf(stdout,"    set time unit to 1.0\n");
        timeUnit = 1;
    }


    /****************************************************************************/
    /*                                                   1. TAPING THE FUNCTION */
    /*                                                      ---> always         */
    fprintf(stdout,"--------------------------------------------------------");
    fprintf(stdout,"\nTASK %d: Taping the function\n",
            taskCount++);

    t00 = myclock();
    /* NOTE: taping will be performed ONCE only */
    depTape = tapingScalarFunction(tag,indeps);
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
    fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit*evalCount,
            (t01-t00));

    /****************************************************************************/
    /*                                                           2. ZOS_FORWARD */
    if (controlParameters[cpZosFW]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: forward(tag, m=1, n=%d, keep, X[n], Y[m])\n",
                taskCount++,indepDim);
        fprintf(stdout,"         ---> zos_forward\n");

        /*----------------------------------------------------------------------*/
        /* NO KEEP */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,1,indepDim,0,indeps,&depTape);
        t01 = myclock();

        fprintf(stdout,"    NO KEEP");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* KEEP */
        t02 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,1,indepDim,1,indeps,&depTape);
        t03 = myclock();

        fprintf(stdout,"    KEEP   ");
        fprintf(stdout,TIMEFORMAT,(t03-t02)*timeUnit,
                (t03-t02)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpZosFW] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
            fprintf(stdout,"    Should be the same values:\n");
            fprintf(stdout,"    (original) %12.8E =? %12.8E (forward from tape)\n",
                    depOrig,depTape);
        }
    }


    /****************************************************************************/
    /*                                                           3. FOS_FORWARD */
    if (controlParameters[cpFosFW]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: forward(tag, m=1, n=%d, d=1, keep, X[n][d+1], Y[d+1])\n",
                taskCount++,indepDim);
        fprintf(stdout,"         ---> fos_forward\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        XPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            XPP[i] = new double[2];
            XPP[i][0] = indeps[i];
            XPP[i][1] = (double)rand();
        }
        YP = new double[2];

        /*----------------------------------------------------------------------*/
        /* NO KEEP */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,1,indepDim,1,0,XPP,YP);
        t01 = myclock();

        fprintf(stdout,"    NO KEEP");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* KEEP */
        t02 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,1,indepDim,1,2,XPP,YP);
        t03 = myclock();

        fprintf(stdout,"    KEEP   ");
        fprintf(stdout,TIMEFORMAT,(t03-t02)*timeUnit,
                (t03-t02)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpFosFW] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
            fprintf(stdout,"    Should be the same values:\n");
            fprintf(stdout,"    (original) %12.8E =? %12.8E (forward from tape)\n",
                    depOrig,YP[0]);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<indepDim; i++)
            delete[] XPP[i];
        delete[] XPP;
        delete[] YP;
    }


    /****************************************************************************/
    /*                                                           4. HOS_FORWARD */
    if (controlParameters[cpHosFW]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: forward(tag, m=1, n=%d, d=%d, keep, X[n][d+1], Y[d+1])\n",
                taskCount++,indepDim,degree);
        fprintf(stdout,"         ---> hos_forward\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        XPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            XPP[i] = new double[1+degree];
            XPP[i][0] = indeps[i];
            for (j=1; j<=degree; j++)
                XPP[i][j] = (double)rand();
        }
        YP = new double[1+degree];

        /*----------------------------------------------------------------------*/
        /* NO KEEP */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,1,indepDim,degree,0,XPP,YP);
        t01 = myclock();

        fprintf(stdout,"    NO KEEP");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* KEEP */
        t02 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,1,indepDim,degree,keep,XPP,YP);
        t03 = myclock();

        fprintf(stdout,"    KEEP   ");
        fprintf(stdout,TIMEFORMAT,(t03-t02)*timeUnit,
                (t03-t02)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpHosFW] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
            fprintf(stdout,"    Should be the same values:\n");
            fprintf(stdout,"    (original) %12.8E =? %12.8E (forward from tape)\n",
                    depOrig,YP[0]);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<indepDim; i++)
            delete[] XPP[i];
        delete[] XPP;
        delete[] YP;
    }


    /****************************************************************************/
    /*                                                           5. FOV_FORWARD */
    if (controlParameters[cpFovFW]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: forward(tag, m=1, n=%d, p=%d, x[n], X[n][p], y[m], Y[m][p])\n",
                taskCount++,indepDim,pFW);
        fprintf(stdout,"         ---> fov_forward\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        XPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            XPP[i] = new double[pFW];
            for (j=0; j<pFW; j++)
                XPP[i][j] = (double)rand();
        }
        YP  = new double[1];
        YPP = new double*[1];
        YPP[0] = new double[pFW];

        /*----------------------------------------------------------------------*/
        /* always NO KEEP */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,1,indepDim,pFW,indeps,XPP,YP,YPP);
        t01 = myclock();

        fprintf(stdout,"  (NO KEEP)");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpFovFW] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<indepDim; i++)
            delete[] XPP[i];
        delete[] XPP;
        delete[] YP;
        delete[] YPP[0];
        delete[] YPP;
    }


    /****************************************************************************/
    /*                                                           6. HOV_FORWARD */
    if (controlParameters[cpHovFW]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: forward(tag, m=1, n=%d, d=%d, p=%d, x[n], X[n][p][d], y[m], Y[m][p][d])\n",
                taskCount++,indepDim,degree,pFW);
        fprintf(stdout,"         ---> hov_forward\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        XPPP = new double**[indepDim];
        for (i=0; i<indepDim; i++) {
            XPPP[i] = new double*[pFW];
            for (j=0; j<pFW; j++) {
                XPPP[i][j] = new double[degree];
                for (k=0; k<degree; k++)
                    XPPP[i][j][k] = (double)rand();
            }
        }
        YP  = new double[1];
        YPPP = new double**[1];
        YPPP[0] = new double*[pFW];
        for (j=0; j<pFW; j++)
            YPPP[0][j] = new double[degree];

        /*----------------------------------------------------------------------*/
        /* always NO KEEP */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,1,indepDim,degree,pFW,indeps,XPPP,YP,YPPP);
        t01 = myclock();

        fprintf(stdout,"  (NO KEEP)");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpHovFW] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<indepDim; i++) {
            for (j=0; j<pFW; j++)
                delete[] XPPP[i][j];
            delete[] XPPP[i];
        }
        delete[] XPPP;
        delete[] YP;
        for (j=0; j<pFW; j++)
            delete[] YPPP[0][j];
        delete[] YPPP[0];
        delete[] YPPP;
    }


    /****************************************************************************/
    /*                                                           7. FOS_REVERSE */
    if (controlParameters[cpFosRV]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: reverse(tag, m=1, n=%d, d=0, u, Z[n])\n",
                taskCount++,indepDim);
        fprintf(stdout,"         ---> fos_reverse\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        ZP = new double[indepDim];
        u  = (double)rand();

        /*----------------------------------------------------------------------*/
        /* Forward with keep*/
        forward(tag,1,indepDim,1,indeps,&depTape);

        /*----------------------------------------------------------------------*/
        /* Reverse */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = reverse(tag,1,indepDim,0,u,ZP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpFosRV] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        delete[] ZP;
    }


    /****************************************************************************/
    /*                                                           8. HOS_REVERSE */
    if (controlParameters[cpHosRV]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: reverse(tag, m=1, n=%d, d=%d, u, Z[n][d+1])\n",
                taskCount++,indepDim,degree);
        fprintf(stdout,"         ---> hos_reverse\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        ZPP = new double*[indepDim];
        for (i=0; i<indepDim; i++)
            ZPP[i] = new double[degree+1];
        u  = (double)rand();
        XPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            XPP[i] = new double[1+degree];
            XPP[i][0] = indeps[i];
            for (j=1; j<=degree; j++)
                XPP[i][j] = (double)rand();
        }
        YP = new double[1+degree];

        /*----------------------------------------------------------------------*/
        /* Forward with keep*/
        forward(tag,1,indepDim,degree,keep,XPP,YP);

        /*----------------------------------------------------------------------*/
        /* Reverse */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = reverse(tag,1,indepDim,degree,u,ZPP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpHosRV] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<indepDim; i++)
            delete[] ZPP[i];
        delete[] ZPP;
        for (i=0; i<indepDim; i++)
            delete[] XPP[i];
        delete[] XPP;
        delete[] YP;
    }


    /****************************************************************************/
    /*                                                           9. FOV_REVERSE */
    if (controlParameters[cpFovRV]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: reverse(tag, m=1, n=%d, d=0, p=%d, U[p], Z[p][n])\n",
                taskCount++,indepDim,pRV);
        fprintf(stdout,"         ---> fov_reverse\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        ZPP = new double*[pRV];
        for (i=0; i<pRV; i++)
            ZPP[i] = new double[indepDim];
        UP = new double[pRV];
        for (i=0; i<pRV; i++)
            UP[i] = (double)rand();

        /*----------------------------------------------------------------------*/
        /* Forward with keep*/
        forward(tag,1,indepDim,1,indeps,&depTape);

        /*----------------------------------------------------------------------*/
        /* Reverse */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = reverse(tag,1,indepDim,0,pRV,UP,ZPP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpFovRV] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<pRV; i++)
            delete[] ZPP[i];
        delete[] ZPP;
        delete[] UP;
    }


    /****************************************************************************/
    /*                                                          10. HOV_REVERSE */
    if (controlParameters[cpHovRV]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: reverse(tag, m=1, n=%d, d=%d, p=%d, U[p], Z[p][n][d+1], nz[p][n])\n",
                taskCount++,indepDim,degree,pRV);
        fprintf(stdout,"         ---> hov_reverse\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        ZPPP = new double**[pRV];
        for (i=0; i<pRV; i++) {
            ZPPP[i] = new double*[indepDim];
            for (j=0; j<indepDim; j++)
                ZPPP[i][j] = new double[degree+1];
        }
        UP = new double[pRV];
        for (i=0; i<pRV; i++)
            UP[i] = (double)rand();
        XPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            XPP[i] = new double[1+degree];
            XPP[i][0] = indeps[i];
            for (j=1; j<=degree; j++)
                XPP[i][j] = (double)rand();
        }
        YP = new double[1+degree];
        nzPP = new short*[pRV];
        for (i=0; i<pRV; i++)
            nzPP[i] = new short[indepDim];

        /*----------------------------------------------------------------------*/
        /* Forward with keep*/
        forward(tag,1,indepDim,degree,keep,XPP,YP);

        /*----------------------------------------------------------------------*/
        /* Reverse  without nonzero pattern*/
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = reverse(tag,1,indepDim,degree,pRV,UP,ZPPP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Reverse  with nonzero pattern*/
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = reverse(tag,1,indepDim,degree,pRV,UP,ZPPP,nzPP);
        t01 = myclock();

        fprintf(stdout,"       (NZ)");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpHovRV] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<pRV; i++) {
            for (j=0; j<indepDim; j++)
                delete[] ZPPP[i][j];
            delete[] ZPPP[i];
            delete[] nzPP[i];
        }
        delete[] ZPPP;
        delete[] nzPP;
        delete[] UP;
        for (i=0; i<indepDim; i++)
            delete[] XPP[i];
        delete[] XPP;
        delete[] YP;
    }


    /****************************************************************************/
    /*                                                             11. FUNCTION */
    if (controlParameters[cpFunction]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: function(tag, m=1, n=%d, X[n], Y[m])\n",
                taskCount++,indepDim);

        /*----------------------------------------------------------------------*/
        /* Function evaluation */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = function(tag,1,indepDim,indeps,&depTape);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpFunction] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
            fprintf(stdout,"    Should be the same values:\n");
            fprintf(stdout,"    (original) %12.8E =? %12.8E (forward from tape)\n",
                    depOrig,depTape);
        }
    }


    /****************************************************************************/
    /*                                                             12. JACOBIAN */
    if (controlParameters[cpJacobian]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: gradient(tag, n=%d, X[n], G[n])\n",
                taskCount++,indepDim);

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        JP = new double[indepDim];

        /*----------------------------------------------------------------------*/
        /* Gradient evaluation */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = gradient(tag,indepDim,indeps,JP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpJacobian] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        delete[] JP;
    }


    /****************************************************************************/
    /*                                                               13. VECJAC */
    if (controlParameters[cpVecJac]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: vec_jac(tag, m=1, n=%d, repeat, X[n], U[m], V[n])\n",
                taskCount++,indepDim);

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        UP = new double[1];
        UP[0] = (double)rand();
        VP = new double[indepDim];

        /*----------------------------------------------------------------------*/
        /* Evaluation without repeat */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = vec_jac(tag,1,indepDim,0,indeps,UP,VP);
        t01 = myclock();

        fprintf(stdout,"(no repeat)");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Evaluation with repeat */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = vec_jac(tag,1,indepDim,1,indeps,UP,VP);
        t01 = myclock();

        fprintf(stdout,"   (repeat)");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpVecJac] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        delete[] UP;
        delete[] VP;
    }


    /****************************************************************************/
    /*                                                               14. JACVEC */
    if (controlParameters[cpJacVec]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: jac_vec(tag, m=1, n=%d, X[n], V[n], U[m])\n",
                taskCount++,indepDim);

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        UP = new double[1];
        VP = new double[indepDim];
        for (i=0; i<indepDim; i++)
            VP[i] = (double)rand();

        /*----------------------------------------------------------------------*/
        /* Evaluation */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = jac_vec(tag,1,indepDim,indeps,VP,UP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpJacVec] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        delete[] UP;
        delete[] VP;
    }


    /****************************************************************************/
    /*                                                              15. HESSIAN */
    if (controlParameters[cpHessian]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: hessian(tag, n=%d, X[n], lower triangle of H[n][n])\n",
                taskCount++,indepDim);

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        HPP = new double*[indepDim];
        for (i=0; i<indepDim; i++)
            HPP[i] = new double[indepDim];

        /*----------------------------------------------------------------------*/
        /* Evaluation */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = hessian(tag,indepDim,indeps,HPP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpHessian] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<indepDim; i++)
            delete[] HPP[i];
        delete[] HPP;
    }


    /****************************************************************************/
    /*                                                              16. HESSVEC */
    if (controlParameters[cpHessVec]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: hess_vec(tag, n=%d, X[n], V[n], W[n])\n",
                taskCount++,indepDim);

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        VP = new double[indepDim];
        for (i=0; i<indepDim; i++)
            VP[i] = (double)rand();
        WP = new double[indepDim];

        /*----------------------------------------------------------------------*/
        /* Evaluation */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = hess_vec(tag,indepDim,indeps,VP,WP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpHessVec] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        delete[] VP;
        delete[] WP;
    }


    /****************************************************************************/
    /*                                                           17. LAGHESSVEC */
    if (controlParameters[cpLagHessVec]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: lagra_hess_vec(tag, m=1, n=%d, X[n], U[m], V[n], W[n])\n",
                taskCount++,indepDim);

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        UP = new double[1];
        UP[0] = (double)rand();
        VP = new double[indepDim];
        for (i=0; i<indepDim; i++)
            VP[i] = (double)rand();
        WP = new double[indepDim];

        /*----------------------------------------------------------------------*/
        /* Evaluation */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = lagra_hess_vec(tag,1,indepDim,indeps,UP,VP,WP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpLagHessVec] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        delete[] VP;
        delete[] WP;
        delete[] UP;
    }


    /****************************************************************************/
    /*                                                               18. TENSOR */
    if (controlParameters[cpTensor]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: tensor_eval(tag, m =1, n=%d, d=%d, p=%d, X[n], tensor[m][dim], S[n][p])\n",
                taskCount++,indepDim,degree, pTR);
        fprintf(stdout,"\n         dim = ((p+d) over d)\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        dim = binomi(pTR+degree,degree);
        TPP = new double*[1];
        TPP[0] = new double[dim];
        SPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            SPP[i] = new double[pTR];
            for (j=0; j<pTR; j++)
                SPP[i][j]=(i==j)?1.0:0.0;
        }

        /*----------------------------------------------------------------------*/
        /* tensor evaluation */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            tensor_eval(tag,1,indepDim,degree,pTR,indeps,TPP,SPP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
    if (controlParameters[cpTensor] > 1) {}

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        delete[] TPP[0];
        delete[] TPP;
        for (i=0; i<indepDim; i++)
            delete[] SPP[i];
        delete[] SPP;
    }


    /****************************************************************************/
    /*                                                       19. INVERSE TENSOR */
    if (controlParameters[cpInvTensor] && (1==indepDim)) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: inverse_tensor_eval(tag, m=n=1, d=%d, p=%d, X[n], tensor[m][dim], S[n][p])\n",
                taskCount++,degree, pTR);
        fprintf(stdout,"\n         dim = ((p+d) over d)\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        dim = binomi(pTR+degree,degree);
        TPP = new double*[1];
        TPP[0] = new double[dim];
        SPP = new double*[1];
        SPP[0] = new double[pTR];
        for (j=0; j<pTR; j++)
            SPP[0][j]=(0==j)?1.0:0.0;

        /*----------------------------------------------------------------------*/
        /* tensor evaluation */
        t00 = myclock();
        for (i=0; i<evalCount; i++)
            inverse_tensor_eval(tag,1,degree,pTR,indeps,TPP,SPP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
    if (controlParameters[cpInvTensor] > 1) {}

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        delete[] TPP[0];
        delete[] TPP;
        delete[] SPP[0];
        delete[] SPP;
    }

    return 1;
}

#undef _SGENMAIN_C_


