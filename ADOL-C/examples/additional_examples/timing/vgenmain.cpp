/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     vgenmain.cpp
 Revision: $Id$
 Contents: Vector Generic Main File: 
       for use with function modules containing several vector
       function examples 
       (e.g. the sinple example in vfunc_simplevec.cpp)

   Each << function module >> contains:
          
     (1) const char* const controlFileName 
     (2) int indepDim; 
     (3) int depDim; 
     (4) void initProblemParameters( void )
     (5) void initIndependents( double* indeps )
     (6) void originalVectorFunction( double* indeps, double* deps )
     (7) void tapingVectorFunction( int tag, double* indeps, double* deps )   
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 
   
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#define _VGENMAIN_C_


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
extern int depDim;

/*--------------------------------------------------------------------------*/
/*                                                  Init Problem Parameters */
extern void initProblemParameters( void );

/*--------------------------------------------------------------------------*/
/*                                                        Initialize indeps */
extern void initIndependents( double* indeps );

/*--------------------------------------------------------------------------*/
/*                                                 Original scalar function */
extern void originalVectorFunction( double* indeps, double* deps );

/*--------------------------------------------------------------------------*/
/*                                                   Taping scalar function */
extern void tapingVectorFunction( int tag, double* indeps, double* deps );


/****************************************************************************/
/*                                                            CONTROL STUFF */
enum controlParameter {
    cpIndepDimension,
    cpDepDimension,
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

    indepDim  = controlParameters[cpIndepDimension];
    depDim    = controlParameters[cpDepDimension];
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
    double* deps   = new double[depDim];
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
            (controlParameters[cpTensor] || controlParameters[cpInvTensor])) {
        fprintf(stdout,"    # of vectors in tensor mode = ? ");
        fscanf(stdin,"%d",&pTR);
        fprintf(stdout,"\n");
    }

    /*------------------------------------------------------------------------*/
    /* Necessary variable */
    double ***XPPP, **XPP;
    double ***YPPP, **YPP, *YP;
    double ***ZPPP, **ZPP, *ZP;
    double          **UPP, *UP;
    double                 *VP;
    double                 *WP;
    double          **JPP;
    short           **nzPP;
    int retVal=0;                 /* return value */
    double t00, t01, t02, t03;  /* time values */
    double          **TPP;
    double          **SPP;
    int dim;


    /****************************************************************************/
    /*                                          0. ORIGINAL FUNCTION EVALUATION */
    /*                                             ---> always                  */
    fprintf(stdout,"\nTASK %d: Original function evaluation\n",
            taskCount++);

    t00 = myclock(1);
    for (i=0; i<evalCount; i++)
        originalVectorFunction(indeps,deps);
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
    fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit*evalCount,
            (t01-t00));

    /****************************************************************************/
    /*                                                           2. ZOS_FORWARD */
    if (controlParameters[cpZosFW]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: forward(tag, m=%d, n=%d, keep, X[n], Y[m])\n",
                taskCount++,depDim,indepDim);
        fprintf(stdout,"         ---> zos_forward\n");

        /*----------------------------------------------------------------------*/
        /* NO KEEP */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,depDim,indepDim,0,indeps,deps);
        t01 = myclock();

        fprintf(stdout,"    NO KEEP");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* KEEP */
        t02 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,depDim,indepDim,1,indeps,deps);
        t03 = myclock();

        fprintf(stdout,"    KEEP   ");
        fprintf(stdout,TIMEFORMAT,(t03-t02)*timeUnit,
                (t03-t02)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpZosFW] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }
    }


    /****************************************************************************/
    /*                                                           3. FOS_FORWARD */
    if (controlParameters[cpFosFW]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: forward(tag, m=%d, n=%d, d=1, keep, X[n][d+1], Y[m][d+1])\n",
                taskCount++,depDim,indepDim);
        fprintf(stdout,"         ---> fos_forward\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        XPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            XPP[i] = new double[2];
            XPP[i][0] = indeps[i];
            XPP[i][1] = (double)rand();
        }
        YPP = new double*[depDim];
        for (i=0; i<depDim; i++)
            YPP[i] = new double[2];

        /*----------------------------------------------------------------------*/
        /* NO KEEP */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,depDim,indepDim,1,0,XPP,YPP);
        t01 = myclock();

        fprintf(stdout,"    NO KEEP");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* KEEP */
        t02 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,depDim,indepDim,1,2,XPP,YPP);
        t03 = myclock();

        fprintf(stdout,"    KEEP   ");
        fprintf(stdout,TIMEFORMAT,(t03-t02)*timeUnit,
                (t03-t02)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpFosFW] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<indepDim; i++)
            delete[] XPP[i];
        delete[] XPP;
        for (i=0; i<depDim; i++)
            delete[] YPP[i];
        delete[] YPP;
    }


    /****************************************************************************/
    /*                                                           4. HOS_FORWARD */
    if (controlParameters[cpHosFW]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: forward(tag, m=%d, n=%d, d=%d, keep, X[n][d+1], Y[m][d+1])\n",
                taskCount++,depDim,indepDim,degree);
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
        YPP = new double*[depDim];
        for (i=0; i<depDim; i++)
            YPP[i] = new double[1+degree];

        /*----------------------------------------------------------------------*/
        /* NO KEEP */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,depDim,indepDim,degree,0,XPP,YPP);
        t01 = myclock();

        fprintf(stdout,"    NO KEEP");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* KEEP */
        t02 = myclock();
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,depDim,indepDim,degree,keep,XPP,YPP);
        t03 = myclock();

        fprintf(stdout,"    KEEP   ");
        fprintf(stdout,TIMEFORMAT,(t03-t02)*timeUnit,
                (t03-t02)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpHosFW] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<indepDim; i++)
            delete[] XPP[i];
        delete[] XPP;
        for (i=0; i<depDim; i++)
            delete[] YPP[i];
        delete[] YPP;
    }


    /****************************************************************************/
    /*                                                           5. FOV_FORWARD */
    if (controlParameters[cpFovFW]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: forward(tag, m=%d, n=%d, p=%d, x[n], X[n][p], y[m], Y[m][p])\n",
                taskCount++,depDim,indepDim,pFW);
        fprintf(stdout,"         ---> fov_forward\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        XPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            XPP[i] = new double[pFW];
            for (j=0; j<pFW; j++)
                XPP[i][j] = (double)rand();
        }
        YP  = new double[depDim];
        YPP = new double*[depDim];
        for (i=0; i<depDim; i++)
            YPP[i] = new double[pFW];

        /*----------------------------------------------------------------------*/
        /* always NO KEEP */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,depDim,indepDim,pFW,indeps,XPP,YP,YPP);
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
        for (i=0; i<depDim; i++)
            delete[] YPP[i];
        delete[] YPP;
    }


    /****************************************************************************/
    /*                                                           6. HOV_FORWARD */
    if (controlParameters[cpHovFW]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: forward(tag, m=%d, n=%d, d=%d, p=%d, x[n], X[n][p][d], y[m], Y[m][p][d])\n",
                taskCount++,depDim,indepDim,degree,pFW);
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
        YP  = new double[depDim];
        YPPP = new double**[depDim];
        for (i=0; i<depDim; i++) {
            YPPP[i] = new double*[pFW];
            for (j=0; j<pFW; j++)
                YPPP[i][j] = new double[degree];
        }

        /*----------------------------------------------------------------------*/
        /* always NO KEEP */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = forward(tag,depDim,indepDim,degree,pFW,indeps,XPPP,YP,YPPP);
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
        for (i=0; i<depDim; i++) {
            for (j=0; j<pFW; j++)
                delete[] YPPP[i][j];
            delete[] YPPP[i];
        }
        delete[] YPPP;
    }


    /****************************************************************************/
    /*                                                           7. FOS_REVERSE */
    if (controlParameters[cpFosRV]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: reverse(tag, m=%d, n=%d, d=0, u[m], Z[n])\n",
                taskCount++,depDim,indepDim);
        fprintf(stdout,"         ---> fos_reverse\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        ZP = new double[indepDim];
        UP = new double[depDim];
        for (i=0; i<depDim; i++)
            UP[i] = (double)rand();

        /*----------------------------------------------------------------------*/
        /* Forward with keep*/
        forward(tag,depDim,indepDim,1,indeps,deps);

        /*----------------------------------------------------------------------*/
        /* Reverse */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = reverse(tag,depDim,indepDim,0,UP,ZP);
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
        delete[] UP;
    }


    /****************************************************************************/
    /*                                                           8. HOS_REVERSE */
    if (controlParameters[cpHosRV]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: reverse(tag, m=%d, n=%d, d=%d, u[m], Z[n][d+1])\n",
                taskCount++,depDim,indepDim,degree);
        fprintf(stdout,"         ---> hos_reverse\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        ZPP = new double*[indepDim];
        for (i=0; i<indepDim; i++)
            ZPP[i] = new double[degree+1];
        UP = new double[depDim];
        for (i=0; i<depDim; i++)
            UP[i] = (double)rand();

        XPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            XPP[i] = new double[1+degree];
            XPP[i][0] = indeps[i];
            for (j=1; j<=degree; j++)
                XPP[i][j] = (double)rand();
        }
        YPP = new double*[depDim];
        for (i=0; i<depDim; i++)
            YPP[i] = new double[1+degree];

        /*----------------------------------------------------------------------*/
        /* Forward with keep*/
        forward(tag,depDim,indepDim,degree,keep,XPP,YPP);

        /*----------------------------------------------------------------------*/
        /* Reverse */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = reverse(tag,depDim,indepDim,degree,UP,ZPP);
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
        for (i=0; i<depDim; i++)
            delete[] YPP[i];
        delete[] YPP;
    }


    /****************************************************************************/
    /*                                                           9. FOV_REVERSE */
    if (controlParameters[cpFovRV]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: reverse(tag, m=%d, n=%d, d=0, p=%d, U[p][m], Z[p][n])\n",
                taskCount++,depDim,indepDim,pRV);
        fprintf(stdout,"         ---> fov_reverse\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        ZPP = new double*[pRV];
        for (i=0; i<pRV; i++)
            ZPP[i] = new double[indepDim];
        UPP = new double*[pRV];
        for (i=0; i<pRV; i++) {
            UPP[i] = new double[depDim];
            for (j=0; j<depDim; j++)
                UPP[i][j] = (double)rand();
        }

        /*----------------------------------------------------------------------*/
        /* Forward with keep*/
        forward(tag,depDim,indepDim,1,indeps,deps);

        /*----------------------------------------------------------------------*/
        /* Reverse */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = reverse(tag,depDim,indepDim,0,pRV,UPP,ZPP);
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
        for (i=0; i<pRV; i++)
            delete[] UPP[i];
        delete[] UPP;
    }


    /****************************************************************************/
    /*                                                          10. HOV_REVERSE */
    if (controlParameters[cpHovRV]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: reverse(tag, m=%d, n=%d, d=%d, p=%d, U[p][m], Z[p][n][d+1], nz[p][n])\n",
                taskCount++,depDim,indepDim,degree,pRV);
        fprintf(stdout,"         ---> hov_reverse\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        ZPPP = new double**[pRV];
        for (i=0; i<pRV; i++) {
            ZPPP[i] = new double*[indepDim];
            for (j=0; j<indepDim; j++)
                ZPPP[i][j] = new double[degree+1];
        }
        UPP = new double*[pRV];
        for (i=0; i<pRV; i++) {
            UPP[i] = new double[depDim];
            for (j=0; j<depDim; j++)
                UPP[i][j] = (double)rand();
        }

        XPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            XPP[i] = new double[1+degree];
            XPP[i][0] = indeps[i];
            for (j=1; j<=degree; j++)
                XPP[i][j] = (double)rand();
        }
        YPP = new double*[depDim];
        for (i=0; i<depDim; i++)
            YPP[i] = new double[1+degree];
        nzPP = new short*[pRV];
        for (i=0; i<pRV; i++)
            nzPP[i] = new short[indepDim];

        /*----------------------------------------------------------------------*/
        /* Forward with keep*/
        forward(tag,depDim,indepDim,degree,keep,XPP,YPP);

        /*----------------------------------------------------------------------*/
        /* Reverse  without nonzero pattern*/
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = reverse(tag,depDim,indepDim,degree,pRV,UPP,ZPPP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Reverse  with nonzero pattern*/
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = reverse(tag,depDim,indepDim,degree,pRV,UPP,ZPPP,nzPP);
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
        for (i=0; i<pRV; i++)
            delete[] UPP[i];
        delete[] UPP;
        for (i=0; i<indepDim; i++)
            delete[] XPP[i];
        delete[] XPP;
        for (i=0; i<depDim; i++)
            delete[] YPP[i];
        delete[] YPP;
    }


    /****************************************************************************/
    /*                                                             11. FUNCTION */
    if (controlParameters[cpFunction]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: function(tag, m=%d, n=%d, X[n], Y[m])\n",
                taskCount++,depDim,indepDim);

        /*----------------------------------------------------------------------*/
        /* Function evaluation */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = function(tag,depDim,indepDim,indeps,deps);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
        if (controlParameters[cpFunction] > 1) {
            fprintf(stdout,"\n    Return value: %d\n",retVal);
        }
    }


    /****************************************************************************/
    /*                                                             12. JACOBIAN */
    if (controlParameters[cpJacobian]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: jacobian(tag, m=%d, n=%d, X[n], J[m][n])\n",
                taskCount++,depDim,indepDim);

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        JPP = new double*[depDim];
        for (i=0; i<depDim; i++)
            JPP[i] = new double[indepDim];

        /*----------------------------------------------------------------------*/
        /* Gradient evaluation */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = jacobian(tag,depDim,indepDim,indeps,JPP);
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
        for (i=0; i<depDim; i++)
            delete[] JPP[i];
        delete[] JPP;
    }


    /****************************************************************************/
    /*                                                               13. VECJAC */
    if (controlParameters[cpVecJac]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: vec_jac(tag, m=%d, n=%d, repeat, X[n], U[m], V[n])\n",
                taskCount++,depDim,indepDim);

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        UP = new double[depDim];
        for (i=0; i<depDim; i++)
            UP[i] = (double)rand();
        VP = new double[indepDim];

        /*----------------------------------------------------------------------*/
        /* Evaluation without repeat */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = vec_jac(tag,depDim,indepDim,0,indeps,UP,VP);
        t01 = myclock();

        fprintf(stdout,"(no repeat)");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Evaluation with repeat */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = vec_jac(tag,depDim,indepDim,1,indeps,UP,VP);
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
        fprintf(stdout,"\nTASK %d: jac_vec(tag, m=%d, n=%d, X[n], V[n], U[m])\n",
                taskCount++,depDim,indepDim);

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        UP = new double[depDim];
        VP = new double[indepDim];
        for (i=0; i<indepDim; i++)
            VP[i] = (double)rand();

        /*----------------------------------------------------------------------*/
        /* Evaluation */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = jac_vec(tag,depDim,indepDim,indeps,VP,UP);
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
    /*                                                           15. LAGHESSVEC */
    if (controlParameters[cpLagHessVec]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: lagra_hess_vec(tag, m=%d, n=%d, X[n], U[m], V[n], W[n])\n",
                taskCount++,depDim,indepDim);

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        UP = new double[depDim];
        for (i=0; i<depDim; i++)
            UP[i] = (double)rand();
        VP = new double[indepDim];
        for (i=0; i<indepDim; i++)
            VP[i] = (double)rand();
        WP = new double[indepDim];

        /*----------------------------------------------------------------------*/
        /* Evaluation */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            retVal = lagra_hess_vec(tag,depDim,indepDim,indeps,UP,VP,WP);
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
    /*                                                               16. TENSOR */
    if (controlParameters[cpTensor]) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: tensor_eval(tag, m =%d, n=%d, d=%d, p=%d, X[n], tensor[m][dim], S[n][p])\n",
                taskCount++,depDim,indepDim,degree, pTR);
        fprintf(stdout,"\n         dim = ((p+d) over d)\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        dim = binomi(pTR+degree,degree);
        TPP = new double*[depDim];
        for (i=0; i<depDim; i++)
            TPP[i] = new double[dim];
        SPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            SPP[i] = new double[pTR];
            for (j=0; j<pTR; j++)
                SPP[i][j]=(i==j)?1.0:0.0;
        }

        /*----------------------------------------------------------------------*/
        /* tensor evaluation */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            tensor_eval(tag,depDim,indepDim,degree,pTR,indeps,TPP,SPP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
    if (controlParameters[cpTensor] > 1) {}

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<depDim; i++)
            delete[] TPP[i];
        delete[] TPP;
        for (i=0; i<indepDim; i++)
            delete[] SPP[i];
        delete[] SPP;
    }


    /****************************************************************************/
    /*                                                       17. INVERSE TENSOR */
    if (controlParameters[cpInvTensor] && (depDim==indepDim)) {
        fprintf(stdout,"--------------------------------------------------------");
        fprintf(stdout,"\nTASK %d: inverse_tensor_eval(tag, m=n=%d, d=%d, p=%d, X[n], tensor[n][dim], S[n][p])\n",
                taskCount++,indepDim,degree, pTR);
        fprintf(stdout,"\n         dim = ((p+d) over d)\n");

        /*----------------------------------------------------------------------*/
        /* Allocation & initialisation of tensors */
        dim = binomi(pTR+degree,degree);
        TPP = new double*[indepDim];
        for (i=0; i<depDim; i++)
            TPP[i] = new double[dim];
        SPP = new double*[indepDim];
        for (i=0; i<indepDim; i++) {
            SPP[i] = new double[pTR];
            for (j=0; j<pTR; j++)
                SPP[i][j]=(i==j)?1.0:0.0;
        }

        /*----------------------------------------------------------------------*/
        /* tensor evaluation */
        t00 = myclock(1);
        for (i=0; i<evalCount; i++)
            inverse_tensor_eval(tag,indepDim,degree,pTR,indeps,TPP,SPP);
        t01 = myclock();

        fprintf(stdout,"           ");
        fprintf(stdout,TIMEFORMAT,(t01-t00)*timeUnit,
                (t01-t00)/evalCount);

        /*----------------------------------------------------------------------*/
        /* Debug infos */
    if (controlParameters[cpInvTensor] > 1) {}

        /*----------------------------------------------------------------------*/
        /* Free tensors */
        for (i=0; i<indepDim; i++)
            delete[] TPP[i];
        delete[] TPP;
        for (i=0; i<indepDim; i++)
            delete[] SPP[i];
        delete[] SPP;
    }

    return 1;
}

#undef _VGENMAIN_C_

