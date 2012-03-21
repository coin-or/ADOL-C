/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     jacpatexam.cpp
 Revision: $Id$
 Contents: example for computation of jacobian sparsity pattern

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>
#include <adolc/adolc_sparse.h>
#include "../clock/myclock.h"

#include <string.h>
#include <math.h>

#include <iostream>
using namespace std;

/****************************************************************************/
/*                                                                  DEFINES */

#define TAG 11


/****************************************************************************/
/*                                                     EVALUATION FUNCTIONS */

/*--------------------------------------------------------------------------*/
const unsigned int N = 5, M = 6;

void eval_small(short tag, double *xp, double *yp) {
    unsigned int i,j;

    trace_on(tag);

    adouble *x,*y;
    x = new adouble[N];
    y = new adouble[M];
    for (i=0;i<N;i++)
        x[i] <<= xp[i];

    int PD1B = __LINE__;

    y[0] = pow(x[0],1) + pow(x[1],2) + pow(x[2],3);
    y[1] = x[0]        * x[1]        / x[2];
    y[2] =                                          asin(x[3]*0.1);
    y[3] = sqrt(x[0])  + sqrt(x[1])  + sqrt(x[2]) + sqrt(x[3])      + sqrt(x[4]);
    y[4] =                                          log(x[3])       + exp(x[4]);
    y[5] =                                          cos(x[3])       - sin(x[4]);

    int PD1E = __LINE__;

    for (j=0;j<M;j++)
        y[j] >>= yp[j];

    delete []x;
    delete []y;

    trace_off(1); // force a numbered tape file to be written

    cout << "\nproblem definition in  "<<__FILE__<<",  lines  "<<PD1B<<" - "<<PD1E<<"\n";
}


/*--------------------------------------------------------------------------*/
const unsigned int NM = 961; // PQ_STRIPMINE_MAX * 8*sizeof(unsigned long int) + 1

void eval_arrow_like_matrix(short tag, double *xp, double *yp) {
    unsigned int i,j;

    trace_on(tag);

    adouble *x,*y;
    x = new adouble[NM];
    y = new adouble[NM];
    for (i=0;i<NM;i++)
        x[i] <<= xp[i];

    int PD2B = __LINE__;

    for (i=0;i<NM;i++) {
        /* dense diagonal and dense last column*/
        y[i] = cos(x[i]) + sin(x[NM-1]);
    }
    for (i=0;i<NM;i++)
        /* dense last row */
        y[NM-1] += sin(x[i]);

    int PD2E = __LINE__;

    for (j=0;j<NM;j++)
        y[j] >>= yp[j];

    delete []x;
    delete []y;

    trace_off(1); // force a numbered tape file to be written

    cout << "\nproblem definition in  "<<__FILE__<<",  lines  "<<PD2B<<" - "<<PD2E<<"\n";
}


/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main(void) {
    short  tag;
    int    ret_c = -1, choice;
    int    oper, op_buf_size, loc_buf_size, con_buf_size, maxlive, deaths;
    unsigned int depen, indep;
    size_t       tape_stats[STAT_SIZE];
    unsigned int    i, j, minnz, maxnz, nz, nzref, nz_rel;
    double z1, z2, t0, t1, t2, t3, t4, t5, t6=0.0, t7=0.0;
    char   outp, full_jac;
    int    precision, width;

    /*--------------------------------------------------------------------------*/
    /*                    variables needed for the Jacobian pattern exploration */

    unsigned int  **jacpat=NULL; // compressed row storage
    double        *base, *value;
    double        basepoint ;
    int           ctrl[3];

    cout << "----------------------------------------------------------------\n";
    cout << "\n                Jacobian Pattern Example\n\n";
    cout << "Tape identification tag ( [-4..-1] for standart examples ) :  ?\b" ;
    cin  >> choice ;

    cout << "\n\nOutput Jacobian pattern? (y/n)  ?\b";
    cin >> outp;

    cout << "\n\nCompare with the full Jacobian calculation? (y/n)  ?\b";
    cin >> full_jac;
    if (( full_jac == 'y' ) || ( full_jac == 'Y'))
        full_jac = 1;
    else
        full_jac = 0;

    cout << "----------------------------------------------------------------\n";

    /*--------------------------------------------------------------------------*/


    if ( choice < 0 ) // Take function in the "eval(...)" routines -------------
    {
        if ( choice > -4 ) {
            base = new double[N];
            for (i=0;i<N;i++)
                base[i] = i+1;

            value = new double[M];

            tag = TAG;
            eval_small(tag, base, value);

            cout << "\n\nCreated ADOL-C tape with identification tag " << tag << ".\n\n";
            cout.flush();


        } else // ( choice == -4 ) -----------------------------------------------
        {

            base = new double[NM];
            for (i=0;i<NM;i++)
                base[i] = i;

            value = new double[NM];

            tag = TAG;
            eval_arrow_like_matrix(tag, base, value);

            cout << "\n\nCreated ADOL-C tape with identification tag " << tag << ".\n\n";
            cout.flush();

        }

        tapestats(tag,tape_stats);              /* Reading of tape statistics */
        indep        = tape_stats[NUM_INDEPENDENTS];
        depen        = tape_stats[NUM_DEPENDENTS];
        op_buf_size  = tape_stats[OP_BUFFER_SIZE];
        loc_buf_size = tape_stats[OP_BUFFER_SIZE];
        con_buf_size = tape_stats[OP_BUFFER_SIZE];
        oper         = tape_stats[NUM_OPERATIONS];
        deaths       = tape_stats[TAY_STACK_SIZE];
        maxlive      = tape_stats[NUM_MAX_LIVES];

        cout<<"\nTape " << tag << ": \n";
        cout<<"  independents              "<<indep<<"\n";
        cout<<"  dependents                "<<depen<<"\n";
        cout<<"  operations                "<<oper<<"\n";
        cout<<"  operations buffer size    "<<op_buf_size<<"\n";
        cout<<"  locations buffer size     "<<loc_buf_size<<"\n";
        cout<<"  constants buffer size     "<<con_buf_size<<"\n";
        cout<<"  maxlive                   "<<maxlive<<"\n";
        cout<<"  valstack size             "<<deaths<<"\n";
        cout<<"\n";

    } else // ( choice >= 0 ) : Take a written tape ------------------------------
    {
        tag = choice;

        cout << "\nproblem definition in  tape "<<tag<<"\n";

        tapestats(tag,tape_stats);
        indep        = tape_stats[NUM_INDEPENDENTS];
        depen        = tape_stats[NUM_DEPENDENTS];
        op_buf_size  = tape_stats[OP_BUFFER_SIZE];
        loc_buf_size = tape_stats[OP_BUFFER_SIZE];
        con_buf_size = tape_stats[OP_BUFFER_SIZE];
        oper         = tape_stats[NUM_OPERATIONS];
        deaths       = tape_stats[TAY_STACK_SIZE];
        maxlive      = tape_stats[NUM_MAX_LIVES];

        cout<<"\nTape " << tag << ": \n";
        cout<<"  independents              "<<indep<<"\n";
        cout<<"  dependents                "<<depen<<"\n";
        cout<<"  operations                "<<oper<<"\n";
        cout<<"  operations buffer size    "<<op_buf_size<<"\n";
        cout<<"  locations buffer size     "<<loc_buf_size<<"\n";
        cout<<"  constants buffer size     "<<con_buf_size<<"\n";
        cout<<"  maxlive                   "<<maxlive<<"\n";
        cout<<"  valstack size             "<<deaths<<"\n";

        cout << "\n\nbasepoint[0.."<< indep <<"] = ?\b";
        cin  >> basepoint;

        base  = new double[indep];
        value = new double[depen];

        for (i=0;i<indep;i++)
            base[i] = basepoint;
    }

    tape_doc(tag,depen,indep,base,value); // write a tape into a tex-file


    /*--------------------------------------------------------------------------*/
    /*                                 Jacobian pattern by index domains, safe */

    cout << "\nJacobian Pattern by index domain propagation, safe ...\n";


    jacpat = new unsigned int* [depen];

    ctrl[0] = 0; // index domain propagation
    ctrl[1] = 0; // automatic mode choice (ignored here)
    ctrl[2] = 0; // safe

    z1 = myclock() ;
    ret_c = jac_pat( tag, depen, indep,  base, jacpat, ctrl);
    z2 = myclock() ;

    if (( outp == 'y' ) || ( outp == 'Y')) {
        for (i=0;i<depen;i++) {
            cout <<"depen["<< i <<"], "<< jacpat[i][0] <<" non-zero entries :\n";
            for (j=1;j<=jacpat[i][0];j++)
                cout << jacpat[i][j] <<"  ";
            cout <<"\n";
        }
        cout.flush();
    }

    t0 = z2 - z1;

    nz = 0;
    minnz = indep;
    maxnz = 0;
    for (i=0;i<depen;i++) {
        nz += jacpat[i][0];
        if ( jacpat[i][0] < minnz )
            minnz = jacpat[i][0];
        if ( jacpat[i][0] > maxnz )
            maxnz = jacpat[i][0];
    }
    nz_rel = (int) ceil (100*nz / ((double)depen*indep));
    cout << nz << " non-zero Jacobian elements  of total " << depen*indep << " elements <= " << nz_rel << "%\n";
    cout << "min " << minnz << " non-zeros per row;    max " << maxnz << " non-zeros per row;\n";
    nzref = nz;

    for (i=0;i<depen;i++)
        myfree1_uint(jacpat[i]);
    delete[] jacpat;
    jacpat=NULL;


    /*--------------------------------------------------------------------------*/
    /*                                 Jacobian pattern by index domains, tight */

    cout << "\nJacobian Pattern by index domain propagation, tight ...\n";

    jacpat = new unsigned int* [depen];

    ctrl[0] = 0; // index domain propagation
    ctrl[1] = 1; // forward (ignored here)
    ctrl[2] = 1; // tight

    z1 = myclock() ;
    ret_c = jac_pat( tag, depen, indep,  base, jacpat, ctrl);
    z2 = myclock() ;

    if (( outp == 'y' ) || ( outp == 'Y')) {
        for (i=0;i<depen;i++) {
            cout <<"depen["<< i <<"], "<< jacpat[i][0] <<" non-zero entries :\n";
            for (j=1;j<=jacpat[i][0];j++)
                cout << jacpat[i][j] <<"  ";
            cout <<"\n";
        }
        cout.flush();
    }

    t1 = z2 - z1;

    nz = 0;
    minnz = indep;
    maxnz = 0;
    for (i=0;i<depen;i++) {
        nz += jacpat[i][0];
        if ( jacpat[i][0] < minnz )
            minnz = jacpat[i][0];
        if ( jacpat[i][0] > maxnz )
            maxnz = jacpat[i][0];
    }
    nz_rel = (int) ceil (100*nz / ((double)depen*indep));
    cout << nz << " non-zero Jacobian elements of total " << depen*indep << " elements <= " << nz_rel << "%\n";
    cout << "min " << minnz << " non-zeros per row;    max " << maxnz << " non-zeros per row;\n";
    if ( nz != nzref )
        cout << "\n\n!!! This method found a different number of non-zeros !!!\n\n";
    cout << "\n\n\n";

    for (i=0;i<depen;i++)
        myfree1_uint(jacpat[i]);
    delete[] jacpat;
    jacpat=NULL;

    /*--------------------------------------------------------------------------*/
    /*                          Jacobian pattern by bit pattern, forward, tight */

    cout << "\nJacobian Pattern by bit pattern propagation, forward, tight ...\n";

    jacpat = new unsigned int* [depen];

    ctrl[0] = 1; // bit pattern propagation
    ctrl[0] = 1; // forward
    ctrl[1] = 1; // tight

    z1 = myclock() ;
    ret_c = jac_pat( tag, depen, indep,  base, jacpat, ctrl);
    z2 = myclock() ;

    if (( outp == 'y' ) || ( outp == 'Y')) {
        for (i=0;i<depen;i++) {
            cout <<"depen["<< i <<"], "<< jacpat[i][0] <<" non-zero entries :\n";
            for (j=1;j<=jacpat[i][0];j++)
                cout << jacpat[i][j] <<"  ";
            cout <<"\n";
        }
        cout.flush();
    }

    t2 = z2 - z1;

    nz = 0;
    minnz = indep;
    maxnz = 0;
    for (i=0;i<depen;i++) {
        nz += jacpat[i][0];
        if ( jacpat[i][0] < minnz )
            minnz = jacpat[i][0];
        if ( jacpat[i][0] > maxnz )
            maxnz = jacpat[i][0];
    }
    nz_rel = (int) ceil (100*nz / ((double)depen*indep));
    cout << nz << " non-zero Jacobian elements of total " << depen*indep << " elements <= " << nz_rel << "%\n";
    cout << "min " << minnz << " non-zeros per row;    max " << maxnz << " non-zeros per row;\n";
    if ( nz != nzref )
        cout << "\n\n!!! This method found a different number of non-zeros !!!\n\n";

    for (i=0;i<depen;i++)
        myfree1_uint(jacpat[i]);
    delete[] jacpat;
    jacpat=NULL;


    /*--------------------------------------------------------------------------*/
    /*                           Jacobian pattern by bit pattern, forward, safe */

    cout << "\nJacobian Pattern by bit pattern propagation, forward, safe ...\n";

    jacpat = new unsigned int* [depen];

    ctrl[0] = 1; // bit pattern propagation
    ctrl[1] = 1; // forward
    ctrl[2] = 0; // safe

    z1 = myclock() ;
    ret_c = jac_pat( tag, depen, indep,  base, jacpat, ctrl);
    z2 = myclock() ;

    if (( outp == 'y' ) || ( outp == 'Y')) {
        for (i=0;i<depen;i++) {
            cout <<"depen["<< i <<"], "<< jacpat[i][0] <<" non-zero entries :\n";
            for (j=1;j<=jacpat[i][0];j++)
                cout << jacpat[i][j] <<"  ";
            cout <<"\n";
        }
        cout.flush();
    }

    t3 = z2 - z1;

    nz = 0;
    minnz = indep;
    maxnz = 0;
    for (i=0;i<depen;i++) {
        nz += jacpat[i][0];
        if ( jacpat[i][0] < minnz )
            minnz = jacpat[i][0];
        if ( jacpat[i][0] > maxnz )
            maxnz = jacpat[i][0];
    }
    nz_rel = (int) ceil (100*nz / ((double)depen*indep));
    cout << nz << " non-zero Jacobian elements of total " << depen*indep << " elements <= " << nz_rel << "%\n";
    cout << "min " << minnz << " non-zeros per row;    max " << maxnz << " non-zeros per row;\n";
    if ( nz != nzref )
        cout << "\n\n!!! This method found a different number of non-zeros !!!\n\n";
    cout << "\n\n\n";

    for (i=0;i<depen;i++)
        myfree1_uint(jacpat[i]);
    delete[] jacpat;
    jacpat=NULL;

    /*--------------------------------------------------------------------------*/
    /*                          Jacobian pattern by bit pattern, reverse, tight */

    cout << "\nJacobian Pattern by bit pattern propagation, reverse, tight ...\n";

    jacpat = new unsigned int* [depen];

    ctrl[0] = 1; // bit pattern propagation
    ctrl[1] = 2; // reverse
    ctrl[2] = 1; // tight

    z1 = myclock() ;
    ret_c = jac_pat( tag, depen, indep,  base, jacpat, ctrl);
    z2 = myclock() ;

    if (( outp == 'y' ) || ( outp == 'Y')) {
        for (i=0;i<depen;i++) {
            cout <<"depen.["<< i <<"], "<< jacpat[i][0] <<" non-zero entries :\n";
            for (j=1;j<=jacpat[i][0];j++)
                cout << jacpat[i][j] <<"  ";
            cout <<"\n";
        }
        cout.flush();
    }

    t4 = z2 - z1;

    nz = 0;
    minnz = indep;
    maxnz = 0;
    for (i=0;i<depen;i++) {
        nz += jacpat[i][0];
        if ( jacpat[i][0] < minnz )
            minnz = jacpat[i][0];
        if ( jacpat[i][0] > maxnz )
            maxnz = jacpat[i][0];
    }
    nz_rel = (int) ceil (100*nz / ((double)depen*indep));
    cout << nz << " non-zero Jacobian elements of total " << depen*indep << " elements <= " << nz_rel << "%\n";
    cout << "min " << minnz << " non-zeros per row;    max " << maxnz << " non-zeros per row;\n";
    if ( nz != nzref )
        cout << "\n\n!!! This method found a different number of non-zeros !!!\n\n";

    for (i=0;i<depen;i++)
        myfree1_uint(jacpat[i]);
    delete[] jacpat;
    jacpat=NULL;


    /*--------------------------------------------------------------------------*/
    /*                           Jacobian pattern by bit pattern, reverse, safe */

    cout << "\nJacobian Pattern by bit pattern propagation, reverse, safe ...\n";

    jacpat = new unsigned int* [depen];

    ctrl[0] = 1; // bit pattern propagation 
    ctrl[1] = 2; // reverse
    ctrl[2] = 0; // safe

    z1 = myclock() ;
    ret_c = jac_pat( tag, depen, indep,  base, jacpat, ctrl);
    z2 = myclock() ;

    if (( outp == 'y' ) || ( outp == 'Y')) {
        for (i=0;i<depen;i++) {
            cout <<"depen.["<< i <<"], "<< jacpat[i][0] <<" non-zero entries :\n";
            for (j=1;j<=jacpat[i][0];j++)
                cout << jacpat[i][j] <<"  ";
            cout <<"\n";
        }
        cout.flush();
    }

    t5 = z2 - z1;

    nz = 0;
    minnz = indep;
    maxnz = 0;
    for (i=0;i<depen;i++) {
        nz += jacpat[i][0];
        if ( jacpat[i][0] < minnz )
            minnz = jacpat[i][0];
        if ( jacpat[i][0] > maxnz )
            maxnz = jacpat[i][0];
    }
    nz_rel = (int) ceil (100*nz / ((double)depen*indep));
    cout << nz << " non-zero Jacobian elements of total " << depen*indep << " elements  <= " << nz_rel << "%\n";
    cout << "min " << minnz << " non-zeros per row;    max " << maxnz << " non-zeros per row;\n";
    if ( nz != nzref )
        cout << "\n\n!!! This method found a different number of non-zeros !!!\n\n";
    cout << "\n\n\n";

    for (i=0;i<depen;i++)
        myfree1_uint(jacpat[i]);
    delete[] jacpat;
    jacpat=NULL;


    /* full Jacobian evaluation -----------------------------------------------*/

    if ( full_jac ) {
        /*---------------------------------------------------------------------*/
        /*                        variables needed for the evaluation routines */

        double **Jac = new double*[depen];
        for (i=0;i<depen;i++)
            Jac[i] = new double[indep];
        double **I = new double*[indep];
        for (i=0;i<indep;i++) {
            I[i] = new double[indep];
            for (j=0;j<indep;j++)
                I[i][j] = 0.0;
            I[i][i] = 1.0;
        }


        /*---------------------------------------------------------------------*/
        /*                full Jacobian evaluation by forward, no strip-mining */

        cout << "\nFull Jacobian evaluation by forward(..), no \"strip-mining\" ...\n";

        z1 = myclock() ;

        forward( tag, depen, indep, indep, base, I, value, Jac);

        z2 = myclock() ;

        t6 = z2 - z1;


        /*---------------------------------------------------------------------*/
        /*    full Jacobian evaluation by the jacobian driver, no strip-mining */

        cout << "\nFull Jacobian evaluation by the jacobian driver, no \"strip-mining\" ...\n";

        z1 = myclock() ;

        jacobian( tag, depen, indep, base, Jac);

        z2 = myclock() ;

        t7 = z2 - z1;

    }


    /*--------------------------------------------------------------------------*/
    /* output of timings */
    width = 8;
    precision = 2;

    cout.setf(ios::fixed,ios::floatfield);
    cout.setf(ios::right,ios::adjustfield);
    cout.precision(precision);

    cout << "\n\n----------------------------------------------------------------\n";
    cout << "\n\nTime to explore the Jacobian Pattern by :\n\n";
    cout << " index domain propagation, safe          :  ";
    cout.width(width);
    cout << t0 << " sec.\n";
    cout << " index domain propagation, tight         :  ";
    cout.width(width);
    cout << t1 << " sec.\n";
    cout << " bit pattern propagation, forward, tight :  ";
    cout.width(width);
    cout << t2 << " sec.\n";
    cout << " index domain propagation, forward, safe :  ";
    cout.width(width);
    cout << t3 << " sec.\n";
    cout << " bit pattern propagation, reverse, tight :  ";
    cout.width(width);
    cout << t4 << " sec.\n";
    cout << " bit pattern propagation, reverse, safe  :  ";
    cout.width(width);
    cout << t5 << " sec.\n\n";
    if ( full_jac ) {
        cout << " full Jacobian evaluation, forward   :  ";
        cout.width(width);
        cout << t6 << " sec.\n";
        cout << " full Jacobian evaluation, jacobian  :  ";
        cout.width(width);
        cout << t7 << " sec.\n";
    }

    if ( ! ( t0 && t1 && t2 && t3 && t4  && t5 ) )
        cout <<"\n! Zero timing due to low problem dimension.\n";

    cout <<"\nOK, done.\n";
    cout.flush();

    return 1;
}
