/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     LUdet.cpp
 Revision: $Id$
 Contents: example for
             * Computation of the determinant of a matrix
               by LU-decomposition of the system matrix without pivoting 
             * application of tapedoc to observe taping of
               the new op_codes for the elementary operations
                  
                     y += x1 * x2;
                     y -= x1 * x2;           

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include "LU.h"


/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() { /*------------------------------------------------------------------------*/
    /* variables */
    const int tag   = 1;                       // tape tag
    const int size  = 5;                       // system size
    const int indep = size*size;               // # of indeps
    const int depen = 1;                       // # of deps

    double  A[size][size], a1[size], a2[size], det; // passive variables
    adouble **AA, *AAp,     Adet;                   // active variables
    double *args  = myalloc1(indep);                // arguments
    double *grad  = myalloc1(indep);                // the gradient
    double **hess = myalloc2(indep,indep);          // the hessian

    int i,j;


    /*------------------------------------------------------------------------*/
    /* Info */
    fprintf(stdout,"DETERMINANT by LU-DECOMPOSITION (ADOL-C Example)\n\n");


    /*------------------------------------------------------------------------*/
    /* Allcoation und initialization of the system matrix */
    AA  = new adouble*[size];
    AAp = new adouble[size*size];
    for (i=0; i<size; i++) {
        AA[i] = AAp;
        AAp += size;
    }
    for(i=0; i<size; i++) {
        a1[i] = i*0.25;
        a2[i] = i*0.33;
    }
    for(i=0; i<size; i++) {
        for(j=0; j<size; j++)
            A[i][j] = a1[i]*a2[j];
        A[i][i] += i+1;
    }


    /*------------------------------------------------------------------------*/
    /* Taping the computation of the determinant */
    trace_on(tag);
    /* marking indeps */
    for(i=0; i<size; i++)
        for(j=0; j<size; j++)
            AA[i][j] <<= (args[i*size+j] = A[i][j]);     // indep. vars
    /* LU-factorization and computation of determinant */
    LUfact(size,AA);
    Adet = AA[0][0];
    for (i=1; i<size; i++)
        Adet *= AA[i][i];
    /* marking deps */
    Adet >>= det;
    trace_off();
    fprintf(stdout," Determinant (original):  %16.4E\n",det);


    /*------------------------------------------------------------------------*/
    /* Recomputation of determinant */
    function(tag,depen,indep,args,&det);
    fprintf(stdout," Determinant (from tape): %16.4E\n",det);


    /*------------------------------------------------------------------------*/
    /* Computation of gradient */
    gradient(tag,indep,args,grad);
    fprintf(stdout," Gradient:\n");
    for (i=0; i<size; i++) {
        for (j=0; j<size; j++)
            fprintf(stdout," %14.6E",grad[i*size+j]);
        fprintf(stdout,"\n");
    }

    /*------------------------------------------------------------------------*/
    /* Computation of hessian */
    hessian(tag,indep,args,hess);
    fprintf(stdout," Part of Hessian:\n");
    for (i=0; i<size; i++) {
        for (j=0; j<size; j++)
            fprintf(stdout," %14.6E",hess[0][i*size+j]);
        fprintf(stdout,"\n");
    }

    /*------------------------------------------------------------------------*/
    /* Tape-documentation */
    tape_doc(tag,depen,indep,args,&det);

    /*------------------------------------------------------------------------*/
    /* Tape statistics */
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

    /*------------------------------------------------------------------------*/
    /* That's it */
    return 1;
}








