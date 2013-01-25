/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     hessmat.cpp
 Revision: $Id$
 Contents: example for testing the routines: 
           hov_wk_forward    ( = Higher Order Vector forward With Keep )
           hos_ov_reverse    ( = Higher Order Scalar reverse over vectors)

 Copyright (c) Andrea Walther, Andreas Kowarz, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <stdlib.h>
#include <iostream>
using namespace std;

/****************************************************************************/
/*                                                                     MAIN */
int main() {
    int i,j,l,m,n,d,q,bd, keep;

    /*--------------------------------------------------------------------------*/
    /* inputs */
    cout << "vector x Hessian x matrix for the function \n\n";
    cout << " y[0] = cos(x[0])* ...*cos(x[n]) \n";
    cout << " y[1] = x[0]^n \n";
    cout << " y[2] = condassign(y[i],y[0]>y[1],y[1],y[0]) \n";
    cout << " y[3] = sin(x[0])+ ...+sin(x[n]) \n";
    cout << " y[4] = exp(x[0])- ...-exp(x[n]) \n";
    cout << " y[5] = pow(y[1],3) \n";
    cout << " y[6] += y[5]*y[4] \n";
    cout << " y[7] -= y[6]*y[5] \n";
    cout << " y[j] = 1/x[0]/ .../x[n], j > 3 \n\n";

    cout << " Number of independents = ?\n ";
    cin >> n;
    cout << " Number of dependents =  ?\n ";
    cin >> m;
    cout << " Degree d (for forward) =  ?\n";
    cin >> d;
    cout << " keep (degree of corresponding reverse = keep-1) =  ?\n";
    cout << "       keep <= d+1 must be valid \n";
    cin >> keep;
    cout << " Number of directions =  ?\n ";
    cin >> q;

    /*--------------------------------------------------------------------------*/
    /* allocations and inits */

    double* xp = new double[n];                      /* passive indeps        */
    double* yp = new double[m];                      /* passive depends       */

    /* vector x Hessian x matrix = Upp x H x XPPP */

    double* Up = myalloc(m);        /* vector on left-hand side                */
    double** Upp = myalloc(m,d+1);   /* vector on left-hand side                */
    double*** Xppp = myalloc(n,q,d); /* matrix on right-hand side             */
    double*** Zppp = myalloc(q,n,d+1); /* result of Up x H x XPPP             */

    double*** Yppp = myalloc(m,q,d); /* results of needed hos_wk_forward      */

    /* check results with usual lagra-Hess-vec  */

    double** Xpp = myalloc(n,d);
    double** V   = myalloc(n,q);
    double** W   = myalloc(q,n);
    double** H   = myalloc(n,n);
    double** Ypp = myalloc(m,d);
    double** Zpp = myalloc(n,d+1);
    /* inits */

    for (l=0; l<d; l++)                    /* first everything is set to zero */
        for (i=0; i<n; i++)
            for (j=0;j<q;j++)
                Xppp[i][j][l] = 0;

    /* now carthesian directions as choosen as    */
    /* matrix on right-hand side of Up x H x XPPP */
    bd = (n<q)?n:q;
    for (j=0;j<bd;j++)
        Xppp[j][j][0] = 1;

    for (i=0; i<m; i++)         /* vector on left-hand side of Up x H x XPPP  */
    {Up[i] = 1;                /* is initialised with 1's                    */
        Upp[i][0] = 1;
        for (j=1;j<=d;j++)
            Upp[i][j] = 0;
    }

    for (i=0; i<n; i++)                    /* first everything is set to zero */
        for (j=0;j<d;j++)
            Xpp[i][j] = 0;
    Xpp[0][0] = 1;                 /* now one carthesian direction as choosen */
    /* as vector for lagra-Hess-vec            */

    for (i=0; i<n; i++)                            /* inits of passive indeps */
        xp[i] = (i+1.0)/(2.0+i);

    for (i=0; i<n; i++) {
        for (j=0;j<q;j++)
            V[i][j] = 0;
        if (i < q)
            V[i][i] = 1;
    }

    /*--------------------------------------------------------------------------*/
    trace_on(1);                                      /* tracing the function */

    adouble* x = new adouble[n];                     /* active indeps         */
    adouble* y = new adouble[m];                     /* active depends        */
    for(i=0;i<m;i++)
        y[i] = 1;
    for (i=0; i<n; i++) {
        x[i] <<= xp[i];
        y[0] *= cos(x[i]);
    }

    for(i=1;i<m;i++)
        for(j=0;j<n;j++) {
            switch (i) {
                case 1 :
                    y[i] *= x[0];
                    break;
                case 2 :
#ifndef ADOLC_ADVANCED_BRANCHING
                    condassign(y[i],adouble(y[0]>y[1]),y[1],y[0]);
#else 
                    condassign(y[i],(y[0]>y[1]),y[1],y[0]);
#endif 
                    break;
                case 3 :
                    y[i] -= sin(x[j]);
                    break;
                case 4 :
                    y[i] -= exp(x[j]);
                    break;
                case 5 :
                    y[5] = pow(y[1],3);
                case 6 :
                    y[6] += y[5]*y[4];
                case 7 :
                    y[7] -= y[6]*y[5];
                default :
                    y[i] /= x[j];
            }
        }
    for (i=0; i<m; i++)
        y[i] >>= yp[i] ;
    trace_off();


    /*--------------------------------------------------------------------------*/
    /* work on the tape */


    /* compute results of lagra_hess_vec */
    /* the following is equal to calls inside of lagra_hess_vec(..) */
    /* direct calls to the basic routines hos_forward and hos_reverse */
    /* seem to be faster than call of lagra_hess_vec(..) */
    /* at least in some of our test cases */

    hos_forward(1,m,n,d,keep,xp,Xpp,yp,Ypp);
    hos_reverse(1,m,n,keep-1,Up,Zpp);

    printf("\n Results of hos_reverse:\n\n");

    for (i=0; i<=d; i++) {
        printf(" d = %d \n",i);
        for (j=0;j<n;j++)
            printf(" %6.3f ",Zpp[j][i]);
        printf("\n");
    }

    /* The new drivers. First, hov_wk_forward(..) is called.
       So far, it was impossible to store the results of 
       a higher-order-vector (=hov) forward in order to perform
       a corresponding reverse sweep (for no particular reason.
       Now we have hov with keep (=wk) and the results needed on
       the way back are stored in a specific tape */

    hov_wk_forward(1,m,n,d,keep,q,xp,Xppp,yp,Yppp);

    /* The corresponding reverse sweep
       So far we had only a higher-order-scalar (=hos, scalar because
       only one vector on the left-hand-side) for a scalar forward
       call.
       Now, we use the stored vector information (= hos vector)
       to compute multiple lagra_hess_vec at once */

    hos_ov_reverse(1,m,n,keep-1,q,Upp,Zppp);

    printf("\n Results of hosv_reverse:\n");

    for (l=0; l<q; l++) {
        for (i=0; i<=d; i++) {
            printf(" d = %d \n",i);
            for (j=0;j<n;j++)
                printf(" %6.3f ",Zppp[l][j][i]);
            printf("\n");
        }
        printf("\n\n");
    }

    if (m==1) {
        printf("hess_mat:\n");
        hess_mat(1,n,q,xp,V,W);
        for (i=0; i<q; i++) {
            for (j=0;j<n;j++)
                printf(" %6.3f ",W[i][j]);
            printf("\n");
        }
        printf("hessian2:\n");
        hessian2(1,n,xp,H);
        for (i=0; i<n; i++) {
            for (j=0;j<n;j++)
                printf(" %6.3f ",H[i][j]);
            printf("\n");
        }
    }

    myfree(Zpp);
    myfree(Ypp);
    myfree(H);
    myfree(W);
    myfree(V);
    myfree(Xpp);
    myfree(Zppp);
    myfree(Yppp);
    myfree(Xppp);
    myfree(yp);
    myfree(xp);
    myfree(Up);
    return 1;
}


/****************************************************************************/
/*                                                               THAT'S ALL */




