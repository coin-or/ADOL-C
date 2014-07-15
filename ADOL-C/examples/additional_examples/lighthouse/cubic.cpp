/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     cubic.cpp
 Revision: $Id$
 Contents: example for cubic lighthouse example of Griewank's Book
           using Cardan's formula with two tapes
  

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
adouble activeCubicLighthouse1( adouble t ) {
    adouble p, q, d, r, u, u1,u2, v, c;
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
    /*---------------------*/
    c += 2.0;
    return c;
}

/****************************************************************************/
/*                                                          ADOUBLE ROUTINE */
adouble activeCubicLighthouse2( adouble t ) {
    adouble p, q, d, r, u, v, c, a, z, b;
    /*---------------------*/
    p = tan(t);
    q = p - 0.2;
    p /= 3.0;
    d = q*q;
    d -= p*p*p;
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
    z += 2.0;
    return z;
}

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
    int i, vc;
    int tag1 = 1, tag2 = 2;
    double z, z1, z2, t, tmin, tmax, tdist, dz;

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
    trace_on(tag1);
    at <<= t;
    az = activeCubicLighthouse1(at);
    az >>= z;
    trace_off();
    trace_on(tag2);
    at <<= t;
    az = activeCubicLighthouse2(at);
    az >>= z;
    trace_off();

    /*--------------------------------------------------------------------------*/
    size_t tape_stats[STAT_SIZE];

    tapestats(tag1,tape_stats);

    fprintf(stdout,"\n    independents            %zu\n",tape_stats[NUM_INDEPENDENTS]);
    fprintf(stdout,"    dependents              %zu\n",tape_stats[NUM_DEPENDENTS]);
    fprintf(stdout,"    operations              %zu\n",tape_stats[NUM_OPERATIONS]);
    fprintf(stdout,"    operations buffer size  %zu\n",tape_stats[OP_BUFFER_SIZE]);
    fprintf(stdout,"    locations buffer size   %zu\n",tape_stats[LOC_BUFFER_SIZE]);
    fprintf(stdout,"    constants buffer size   %zu\n",tape_stats[VAL_BUFFER_SIZE]);
    fprintf(stdout,"    maxlive                 %zu\n",tape_stats[NUM_MAX_LIVES]);
    fprintf(stdout,"    valstack size           %zu\n\n",tape_stats[TAY_STACK_SIZE]);

    tapestats(tag2,tape_stats);

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
        function(tag1,1,1,&t,&z1);
        function(tag2,1,1,&t,&z2);
        if (!(z1==z1)) // check for NaN
        { gradient(tag2,1,&t,&dz);
            fprintf(stdout,"%e %e %e\n",t,z2,dz);
        } else {
            gradient(tag1,1,&t,&dz);
            fprintf(stdout,"%e %e %e\n",t,z1,dz);
        }
        t += tdist;
    }

    /*--------------------------------------------------------------------------*/
    return 1;
}




