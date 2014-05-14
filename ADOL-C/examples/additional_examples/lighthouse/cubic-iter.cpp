/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     cubic-iter-2.cpp
 Revision: $Id$
 Contents: example for cubic lighthouse example of Griewank's Book
            using iterative solvers

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


/****************************************************************************/
/*                                                          ADOUBLE ROUTINE */
adouble g( adouble z, adouble t ) {
    adouble v1, v15, v2, v3, res;
    v1 = z - 2.0;
    v15 = v1*v1*v1;
    v15 += 0.4;
    v2 = tan(t);
    v15 -= z * v2;
    v3 = 3.0*v1*v1-v2;
    v3 = fabs(v3);
    res = z - v15/v3;
    return res;
}

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
    int i, j, vc, ic;
    int tag = 1;
    double z, dz, z0, dz0, t, tmin, tmax, tdist;
    double x[2], gradg[2];

    /*--------------------------------------------------------------------------*/
    /* Preparation */
    fprintf(stdout,"CUBIC LIGHTHOUSE Using ITERATION (ADOL-C Example)\n\n");
    tmin = 0.1;
    tmax = 0.3;
    z0 = 2.1;
    dz0 = 0.0;
    fprintf(stdout,"How many values = ? \n");
    scanf("%d",&vc);
    fprintf(stdout,"How many iterations = ? \n");
    scanf("%d",&ic);


    /*--------------------------------------------------------------------------*/
    /* 0. time (taping) */
    t = 0.1;
    trace_on(tag);
    adouble az,at;
    az <<= z0;
    at <<= t;
    az = g(az,at);
    az >>= z;
    trace_off();

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
    tdist = (tmax-tmin)/((double) (vc-1));
    x[1] = tmin;
    for (i=0; i<vc; i++) {
        fprintf(stdout,"%e",x[1]);
        x[0] = z0;
        dz = dz0;
        for (j=0; j<ic; j++) {
            function(tag,1,2,x,&z);
            gradient(tag,2,x,gradg);
            x[0] = z;
            dz = gradg[0]*dz + gradg[1];
            if (!(j%2)) fprintf(stdout," %e %e",x[0],dz);
        }
        fprintf(stdout,"\n");
        x[1] += tdist;
    }


    /*--------------------------------------------------------------------------*/
    return 1;
}
