/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     cubic-2.cpp
 Revision: $Id$
 Contents: example for cubic lighthouse example of Griewank's Book
            using Cardan's formula with conditionals

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
#define PI 3.1415926536


/****************************************************************************/
/*                                                          ADOUBLE ROUTINE */
adouble activeCubicLighthouse( adouble t ) {
    adouble p, q, d, r, u, u1, u2, v, a, b, c, z;
    /*---------------------*/
    p = tan(t);
    q = p - 0.2;
    p /= 3.0;
    d = q*q;
    d -= p*p*p;
    /* 1. branch ----------*/
    r = sqrt(d);
    u = q + r;
    u1 = pow(fabs(u),1.0/3.0);
    u2 = -u1;
    condassign(u,u,u1,u2);
    v = q - r;
    u1 = pow(fabs(v),1.0/3.0);
    u2 = -u1;
    condassign(v,v,u1,u2);
    c = u + v;
    /* 2. branch ----------*/
    p = fabs(p);
    p = sqrt(p);
    q /= p*p*p;
    a = acos(q);
    a /= 3.0;
    z = cos(a);
    b = a + PI/3.0;
    b = -cos(b);
    z = fmin(z,b);
    b = a - PI/3.0;
    b = -cos(b);
    z = fmin(z,b);
    z = 2.0*z*p;
    /*---------------------*/
    condassign(z,d,c);
    z += 2.0;
    return z;
}

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
    int i, vc;
    int tag = 1;
    double z, t, tmin, tmax, tdist, dz;

    /*--------------------------------------------------------------------------*/
    /* Preparation */
    fprintf(stdout,"CUBIC LIGHTHOUSE Using CARDAN (ADOL-C Example)\n\n");
    tmin = 0.15;
    tmax = 0.24;
    fprintf(stdout,"How many values = ? \n");
    scanf("%d",&vc);

    /*--------------------------------------------------------------------------*/
    t = 0.1;
    adouble az,at;
    trace_on(tag);
    at <<= t;
    az = activeCubicLighthouse(at);
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
    t = tmin;
    for (i=0; i<vc; i++) {
        function(tag,1,1,&t,&z);
        gradient(tag,1,&t,&dz);
        fprintf(stdout,"%e %e %e\n",t,z,dz);
        t += tdist;
    }

    /*--------------------------------------------------------------------------*/
    return 1;
}




