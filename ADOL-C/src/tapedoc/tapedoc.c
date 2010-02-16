/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     tapedoc/tapedoc.c
 Revision: $Id$
 Contents: Routine tape_doc(..) writes the taped operations in LaTeX-code 
           to the file tape_doc.tex
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#include <tapedoc/tapedoc.h>
#include <oplate.h>
#include <taping_p.h>
#include <adalloc.h>

#include <math.h>
#include <string.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                                   MACROS */
#define computenumbers true


/****************************************************************************/
/*                                                         STATIC VARIABLES */

/*--------------------------------------------------------------------------*/
static short tag;

static int op_cnt;
static int rev_op_cnt;
static int pagelength;
static FILE *fp;

static char baseName[]="tape_";
static char extension[]=".tex";

/*--------------------------------------------------------------------------*/
/* operation names */
static char* a[] =  {  "death not",
                       "assign ind",
                       "assign dep",
                       "assign a",
                       "assign d",
                       "eq plus d",
                       "eq plus a",
                       "eq min d",
                       "eq min a",
                       "eq mult d",
                       "eq mult a",
                       "plus a a",
                       "plus d a",
                       "min a a",
                       "min d a",
                       "mult a a",
                       "mult d a",
                       "div a a",
                       "div d a",
                       "exp op",
                       "cos op",
                       "sin op",
                       "atan op",
                       "log op",
                       "pow op",
                       "asin op",
                       "acos op",
                       "sqrt op",
                       "asinh_op",
                       "acosh_op",
                       "atanh_op",
                       "gen quad",
                       "end of tape",
                       "start of tape",
                       "end of op",
                       "end of int",
                       "end of val",
                       "cond assign $\\longrightarrow$",
                       "cond assign s $\\longrightarrow$",
                       "take stock op",
                       "assign d one",
                       "assign d zero",
                       "incr a",
                       "decr a",
                       "neg sign a",
                       "pos sign a",
                       "min op",
                       "abs val",
                       "eq zero",
                       "neq zero",
                       "le zero",
                       "gt zero",
                       "ge zero",
                       "lt zero",
                       "eq plus prod",
                       "eq min prod",
                       "erf op",
                       "ceil op",
                       "floor op",
                       "extern fctn"
                       "ignore_me"
                    };

/****************************************************************************/
/*                                                     LOCAL WRITE ROUTINES */

/*--------------------------------------------------------------------------*/
void filewrite_start( int opcode ) {
    char *fileName;
    int num;

    fileName=(char *)malloc(sizeof(char)*(9+sizeof(tag)*8+2));
    if (fileName==NULL) fail(ADOLC_MALLOC_FAILED);
    strncpy(fileName, baseName, strlen(baseName));
    num=sprintf(fileName+strlen(baseName), "%d", tag);
    strncpy(fileName+strlen(baseName)+num, extension, strlen(extension));
    fileName[strlen(baseName)+num+strlen(extension)]=0;
    if ((fp = fopen(fileName,"w")) == NULL) {
        fprintf(DIAG_OUT,"cannot open file !\n");
        exit(1);
    }
    fprintf(fp,"\\documentclass{article}\n");
    fprintf(fp,"\\headheight0cm\n");
    fprintf(fp,"\\headsep-1cm\n");
    fprintf(fp,"\\textheight25cm\n");
    fprintf(fp,"\\oddsidemargin-1cm\n");
    fprintf(fp,"\\topmargin0cm\n");
    fprintf(fp,"\\textwidth18cm\n");
    fprintf(fp,"\\begin{document}\n");
    fprintf(fp,"\\tiny\n");
#ifdef computenumbers
    fprintf(fp,"\\begin{tabular}{|r|r|r|l|r|r|r|r||r|r||r|r|r|r|} \\hline \n");
    fprintf(fp," & & code & op & loc & loc & loc & loc & double & double & value & value & value & value \\\\ \\hline \n");
    fprintf(fp," %i & start of tape & & & & & & & & & &  \\\\ \\hline \n",opcode);
#else
    fprintf(fp,"\\begin{tabular}{|r|r|r|l|r|r|r|r||r|r|} \\hline \n");
    fprintf(fp," & & code & op & loc & loc & loc & loc & double & double \\\\ \\hline \n");
    fprintf(fp," %i & start of tape & & & & & & & \\\\ \\hline \n",opcode);
#endif
    pagelength = 0;
}

/****************************************************************************/
/* filewrite( opcode number, number locations, locations, values,           */
/*            number constants, constants )                                 */
/****************************************************************************/
void filewrite( unsigned short opcode, int nloc, int *loc,
                double *val,int ncst, double* cst) {
    int i;

    ++op_cnt;
    --rev_op_cnt;

    if (pagelength == 100) { /* 101 lines per page */
        fprintf(fp,"\\end{tabular}\\\\\n");
        fprintf(fp,"\\newpage\n");
#ifdef computenumbers
        fprintf(fp,"\\begin{tabular}{|r|r|r|l|r|r|r|r||r|r||r|r|r|r|} \\hline \n");
        fprintf(fp," & & code & op & loc & loc & loc & loc & double & double & value & value & value & value \\\\ \\hline \n");
#else
        fprintf(fp,"\\begin{tabular}{|r|r|r|l|r|r|r|r||r|r|} \\hline \n");
        fprintf(fp," & & code & op & loc & loc & loc & loc & double & double \\\\ \\hline \n");
#endif
        pagelength=-1;
    }

    /* write opcode counters and  number */
    fprintf(fp,"%i & %i & %i & ",op_cnt, rev_op_cnt, opcode);

    /* write opcode name if available */
    i=0;
    while (a[opcode][i]) {
        fprintf(fp,"%c",a[opcode][i]);
        i++;
    }

    /* write locations (max 4) right-justified */
    fprintf(fp," &");
    if (opcode!=ext_diff) { /* default */
        for(i=0; i<(4-nloc); i++)
            fprintf(fp," &");
        for(i=0; i<nloc; i++)
            fprintf(fp," %i &",loc[i]);
    } else { /* ext_diff */
        fprintf(fp," fctn %i &",loc[0]);
        for(i=1; i<(4-nloc); i++)
            fprintf(fp," &");
        for(i=1; i<nloc; i++)
            fprintf(fp," %i &",loc[i]);
    }

    /* write values */
#ifdef computenumbers /* values + constants */
    /* constants (max 2) */
    if (opcode==ext_diff)
        nloc=0;
    for(i=0; i<(2-ncst); i++)
        fprintf(fp," &");
    for(i=0; i<ncst; i++)
        fprintf(fp,"$ %e $&",cst[i]);
    /* values (max 4) */
    if (nloc) {
        for(i=0; i<(4-nloc); i++)
            fprintf(fp," &");
        for(i=0; i<nloc-1; i++)
            fprintf(fp,"$ %e $&",val[i]);
        fprintf(fp,"$ %e $",val[nloc-1]);
    } else {
        for(i=0; i<3; i++)
            fprintf(fp," &");
        fprintf(fp," ");
    }
#else /* constants only */
    /* constants (max 2) */
    if (ncst) {
        for(i=0; i<(2-ncst); i++)
            fprintf(fp," &");
        for(i=0; i<ncst-1; i++)
            fprintf(fp,"$ %e $ &",cst[i]);
        fprintf(fp,"$ %e $",val[ncst-1]);
    } else {
        fprintf(fp," &");
        fprintf(fp," ");
    }
#endif

    fprintf(fp,"\\\\ \\hline \n"); /* end line */
    pagelength++;
}

/*--------------------------------------------------------------------------*/
void filewrite_end( int opcode ) {
    ++op_cnt;
    --rev_op_cnt;
#ifdef computenumbers
  fprintf(fp," %i & %i & %i & end of tape & & & & & & & & & &  \\\\ \\hline \n",op_cnt,rev_op_cnt, opcode);
#else
    fprintf(fp," %i & %i & %i & end of tape & & & & & & & \\\\ \\hline \n",op_cnt,rev_op_cnt,opcode);
#endif
    fprintf(fp,"\\end{tabular}");
    fprintf(fp,"\\end{document}");
    fclose(fp);
}


/****************************************************************************/
/*                                                             NOW THE CODE */
void tape_doc(short tnum,         /* tape id */
              int depcheck,       /* consistency chk on # of dependents */
              int indcheck,       /* consistency chk on # of independents */
              double *basepoint,  /* independent variable values */
              double *valuepoint) /* dependent variable values */
{
    /****************************************************************************/
    /*                                                            ALL VARIABLES */
    unsigned char operation;

    locint size = 0;
    locint res  = 0;
    locint arg  = 0;
    locint arg1 = 0;
    locint arg2 = 0;

    double coval = 0, *d = 0;

    int indexi = 0, indexd = 0;

    /* loop indices */
    int  l;

    /* Taylor stuff */
    double *dp_T0;

    /* interface temporaries */
    int loc_a[4];
    double val_a[4], cst_d[2];

    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /****************************************************************************/
    /*                                                                    INITs */

    init_for_sweep(tnum);
    tag = tnum;

    if ((depcheck != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS]) ||
            (indcheck != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS]) ) {
        fprintf(DIAG_OUT,"ADOL-C error: Tape_doc on tape %d  aborted!\n",tag);
        fprintf(DIAG_OUT,"Number of dependent (%d) and/or independent (%d) "
                "variables passed to Tape_doc is\ninconsistant with "
                "number recorded on tape %d (%d:%d)\n", depcheck,
                indcheck, tag, ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS],
                ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS]);
        exit (-1);
    }

    /* globals */
    op_cnt=0;
    rev_op_cnt=ADOLC_CURRENT_TAPE_INFOS.stats[NUM_OPERATIONS];

    dp_T0 = myalloc1(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES]);

    operation=get_op_f();
    while (operation !=end_of_tape) {
        switch (operation) {

                /****************************************************************************/
                /*                                                                  MARKERS */

                /*--------------------------------------------------------------------------*/
            case end_of_op:                                          /* end_of_op */
                filewrite(operation,0,loc_a,val_a,0,cst_d);
                get_op_block_f();
                operation=get_op_f();
                /* Skip next operation, it's another end_of_op */
                break;

                /*--------------------------------------------------------------------------*/
            case end_of_int:                                        /* end_of_int */
                filewrite(operation,0,loc_a,val_a,0,cst_d);
                get_loc_block_f();
                break;

                /*--------------------------------------------------------------------------*/
            case end_of_val:                                        /* end_of_val */
                filewrite(operation,0,loc_a,val_a,0,cst_d);
                get_val_block_f();
                break;

                /*--------------------------------------------------------------------------*/
            case start_of_tape:                                  /* start_of_tape */
                filewrite_start(operation);
                break;

                /*--------------------------------------------------------------------------*/
            case end_of_tape:                                      /* end_of_tape */
                break;


                /****************************************************************************/
                /*                                                               COMPARISON */

                /*--------------------------------------------------------------------------*/
            case eq_zero  :                                            /* eq_zero */
            case neq_zero :                                           /* neq_zero */
            case le_zero  :                                            /* le_zero */
            case gt_zero  :                                            /* gt_zero */
            case ge_zero  :                                            /* ge_zero */
            case lt_zero  :                                            /* lt_zero */
                arg  = get_locint_f();
                loc_a[0] = arg;
#ifdef computenumbers
                val_a[0] = dp_T0[arg];
#endif
                filewrite(operation,1,loc_a,val_a,0,cst_d);
                break;


                /****************************************************************************/
                /*                                                              ASSIGNMENTS */

                /*--------------------------------------------------------------------------*/
            case assign_a:           /* assign an adouble variable an    assign_a */
                /* adouble value. (=) */
                arg = get_locint_f();
                res = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res]= dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case assign_d:            /* assign an adouble variable a    assign_d */
                /* double value. (=) */
                res  = get_locint_f();
                cst_d[0]=get_val_f();
                loc_a[0]=res;
#ifdef computenumbers
                dp_T0[res]= cst_d[0];
                val_a[0]=dp_T0[res];
#endif
                filewrite(operation,1,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case assign_d_one:    /* assign an adouble variable a    assign_d_one */
                /* double value. (1) (=) */
                res  = get_locint_f();
                loc_a[0]=res;
#ifdef computenumbers
                dp_T0[res]= 1.0;
                val_a[0]=dp_T0[res];
#endif
                filewrite(operation,1,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case assign_d_zero:  /* assign an adouble variable a    assign_d_zero */
                /* double value. (0) (=) */
                res  = get_locint_f();
                loc_a[0]=res;
#ifdef computenumbers
                dp_T0[res]= 0.0;
                val_a[0]=dp_T0[res];
#endif
                filewrite(operation,1,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case assign_ind:       /* assign an adouble variable an    assign_ind */
                /* independent double value (<<=) */
                res  = get_locint_f();
                loc_a[0]=res;
#ifdef computenumbers
                dp_T0[res]= basepoint[indexi];
                cst_d[0]= basepoint[indexi];
                val_a[0]=dp_T0[res];
                filewrite(operation,1,loc_a,val_a,1,cst_d);
#else
                filewrite(operation,1,loc_a,val_a,0,cst_d);
#endif
                indexi++;
                break;

                /*--------------------------------------------------------------------------*/
            case assign_dep:           /* assign a float variable a    assign_dep */
                /* dependent adouble value. (>>=) */
                res = get_locint_f();
                loc_a[0]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[res];
                valuepoint[indexd++]=dp_T0[res];
#endif
                filewrite(operation,1,loc_a,val_a,0,cst_d);
                break;


                /****************************************************************************/
                /*                                                   OPERATION + ASSIGNMENT */

                /*--------------------------------------------------------------------------*/
            case eq_plus_d:            /* Add a floating point to an    eq_plus_d */
                /* adouble. (+=) */
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                dp_T0[res] += coval;
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,1,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_plus_a:             /* Add an adouble to another    eq_plus_a */
                /* adouble. (+=) */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res]+= dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_plus_prod:    /* Add an product to an            eq_plus_prod */
                /* adouble. (+= x1*x2) */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res] += dp_T0[arg1]*dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_d:       /* Subtract a floating point from an    eq_min_d */
                /* adouble. (-=) */
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                dp_T0[res] -= coval;
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,1,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_a:        /* Subtract an adouble from another    eq_min_a */
                /* adouble. (-=) */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res]-= dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_prod:     /* Subtract an product from an      eq_min_prod */
                /* adouble. (+= x1*x2) */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res] -= dp_T0[arg1]*dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_d:              /* Multiply an adouble by a    eq_mult_d */
                /* flaoting point. (*=) */
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                dp_T0[res] *= coval;
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,1,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_a:       /* Multiply one adouble by another    eq_mult_a */
                /* (*=) */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res]*= dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case incr_a:                        /* Increment an adouble    incr_a */
                res = get_locint_f();
                loc_a[0] = res;
#ifdef computenumbers
                dp_T0[res]++;
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,1,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case decr_a:                        /* Increment an adouble    decr_a */
                res = get_locint_f();
                loc_a[0] = res;
#ifdef computenumbers
                dp_T0[res]--;
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,1,loc_a,val_a,0,cst_d);
                break;


                /****************************************************************************/
                /*                                                        BINARY OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case plus_a_a:                 /* : Add two adoubles. (+)    plus a_a */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res]=dp_T0[arg1]+dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case plus_d_a:             /* Add an adouble and a double    plus_d_a */
                /* (+) */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res]= dp_T0[arg] + coval;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case min_a_a:              /* Subtraction of two adoubles     min_a_a */
                /* (-) */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res]=dp_T0[arg1]-dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case min_d_a:                /* Subtract an adouble from a    min_d_a */
                /* double (-) */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = coval - dp_T0[arg];
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case mult_a_a:               /* Multiply two adoubles (*)    mult_a_a */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res]=dp_T0[arg1]*dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case mult_d_a:         /* Multiply an adouble by a double    mult_d_a */
                /* (*) */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = coval * dp_T0[arg];
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case div_a_a:           /* Divide an adouble by an adouble    div_a_a */
                /* (/) */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res]=dp_T0[arg1]/dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case div_d_a:             /* Division double - adouble (/)    div_d_a */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = coval / dp_T0[arg];
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,1,cst_d);
                break;


                /****************************************************************************/
                /*                                                         SIGN  OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case pos_sign_a:                                        /* pos_sign_a */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res]= dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case neg_sign_a:                                        /* neg_sign_a */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res]= -dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,0,cst_d);
                break;


                /****************************************************************************/
                /*                                                         UNARY OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case exp_op:                          /* exponent operation    exp_op */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res]= exp(dp_T0[arg]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case sin_op:                              /* sine operation    sin_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                /* olvo 980923 changed order to allow x=sin(x) */
                val_a[0]=dp_T0[arg1];
                dp_T0[arg2]= cos(dp_T0[arg1]);
                dp_T0[res] = sin(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case cos_op:                            /* cosine operation    cos_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                /* olvo 980923 changed order to allow x=cos(x) */
                val_a[0]=dp_T0[arg1];
                dp_T0[arg2]= sin(dp_T0[arg1]);
                dp_T0[res] = cos(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case atan_op:                                              /* atan_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = atan(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case asin_op:                                              /* asin_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = asin(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case acos_op:                                              /* acos_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = acos(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

#ifdef ATRIG_ERF

                /*--------------------------------------------------------------------------*/
            case asinh_op:                                            /* asinh_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = asinh(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case acosh_op:                                           /* acosh_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = acosh(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case atanh_op:                                            /* atanh_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = atanh(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case erf_op:                                                /* erf_op */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = erf(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,0,cst_d);
                break;

#endif
                /*--------------------------------------------------------------------------*/
            case log_op:                                                /* log_op */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res]= log(dp_T0[arg]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case pow_op:                                                /* pow_op */
                arg  = get_locint_f();
                res  = get_locint_f();
                coval   = get_val_f();
                cst_d[0]=coval;
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res] = pow(dp_T0[arg],coval);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case sqrt_op:                                              /* sqrt_op */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                dp_T0[res]= sqrt(dp_T0[arg]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case gen_quad:                                            /* gen_quad */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                cst_d[0] = get_val_f();
                cst_d[1] = get_val_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef computenumbers
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = cst_d[1];
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,2,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case min_op:                                                /* min_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg1;
                loc_a[1] = arg2;
                loc_a[2] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                val_a[0] = dp_T0[arg1];
                val_a[1] = dp_T0[arg2];
                if (dp_T0[arg1] > dp_T0[arg2])
                    dp_T0[res] = dp_T0[arg2];
                else
                    dp_T0[res] = dp_T0[arg1];
                val_a[2] = dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case abs_val:                                              /* abs_val */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = fabs(dp_T0[arg]);
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case ceil_op:                                              /* ceil_op */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = ceil(dp_T0[arg]);
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case floor_op:                 /* Compute ceil of adouble    floor_op */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef computenumbers
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = floor(dp_T0[arg]);
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,1,cst_d);
                break;


                /****************************************************************************/
                /*                                                             CONDITIONALS */

                /*--------------------------------------------------------------------------*/
            case cond_assign:                                      /* cond_assign */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0]=arg;
                loc_a[1]=arg1;
                loc_a[2]=arg2 ;
                loc_a[3]=res;
                cst_d[0]=coval;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                val_a[1]=dp_T0[arg1];
                val_a[2]=dp_T0[arg2];
                if (dp_T0[arg]>0)
                    dp_T0[res]=dp_T0[arg1];
                else
                    dp_T0[res]=dp_T0[arg2];
                val_a[3]=dp_T0[res];
#endif
                filewrite(operation,4,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case cond_assign_s:                                  /* cond_assign_s */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0]=arg;
                loc_a[1]=arg1;
                loc_a[2]=res;
                cst_d[0]=coval;
#ifdef computenumbers
                val_a[0]=dp_T0[arg];
                val_a[1]=dp_T0[arg1];
                if (dp_T0[arg]>0)
                    dp_T0[res]=dp_T0[arg1];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,3,loc_a,val_a,1,cst_d);
                break;


                /****************************************************************************/
                /*                                                          REMAINING STUFF */

                /*--------------------------------------------------------------------------*/
            case take_stock_op:                                  /* take_stock_op */
                size = get_locint_f();
                res  = get_locint_f();
                d    = get_val_v_f(size);
                loc_a[0] = size;
                loc_a[1] = res;
                cst_d[0] = d[0];
#ifdef computenumbers
                for (l=0; l<size; l++)
                    dp_T0[res+l] = d[l];
                val_a[0] = make_nan();
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case death_not:                                          /* death_not */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                filewrite(operation,2,loc_a,val_a,0,cst_d);
                break;

                /****************************************************************************/
            case ext_diff:
                loc_a[0] = get_locint_f() + 1; /* index */
                loc_a[1] = get_locint_f(); /* n */
                loc_a[2] = get_locint_f(); /* m */
                loc_a[3] = get_locint_f(); /* xa[0].loc */
                loc_a[3] = get_locint_f(); /* ya[0].loc */
                loc_a[3] = get_locint_f(); /* dummy */
                filewrite(operation, 3, loc_a, val_a, 0, cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            default:                                                   /* default */
                /* Die here, we screwed up */
                fprintf(DIAG_OUT,"ADOL-C error: Fatal error in tape_doc for op %d\n",
                        operation);
                break;

        } /* endswitch */

        /* Read the next operation */
        operation=get_op_f();
    }  /* endwhile */

    if (operation == end_of_tape) {
        filewrite_end(operation);
    };

    free(dp_T0);
    dp_T0 = NULL;

    end_sweep();
} /* end tape_doc */


/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS
