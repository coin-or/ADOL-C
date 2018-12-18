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

#include <adolc/tapedoc/tapedoc.h>
#include "oplate.h"
#include "taping_p.h"
#include <adolc/adalloc.h>
#include "dvlparms.h"

#include <math.h>
#include <string.h>

#ifdef ADOLC_AMPI_SUPPORT
#include "ampi/ampi.h"
#include "ampi/tape/support.h"
#endif

BEGIN_C_DECLS

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
        adolc_exit(1,"",__func__,__FILE__,__LINE__);
    }
    free((void*)fileName);
    fprintf(fp,"\\documentclass{article}\n");
    fprintf(fp,"\\headheight0cm\n");
    fprintf(fp,"\\headsep-1cm\n");
    fprintf(fp,"\\textheight25cm\n");
    fprintf(fp,"\\oddsidemargin-1cm\n");
    fprintf(fp,"\\topmargin0cm\n");
    fprintf(fp,"\\textwidth18cm\n");
    fprintf(fp,"\\begin{document}\n");
    fprintf(fp,"\\tiny\n");
#ifdef ADOLC_TAPE_DOC_VALUES
    fprintf(fp,"\\begin{tabular}{|r|r|r|l|r|r|r|r||r|r||r|r|r|r|} \\hline \n");
    fprintf(fp," & & code & op & loc & loc & loc & loc & double & double & value & value & value & value \\\\ \\hline \n");
    fprintf(fp," & & %i & start of tape & & & & & & & & & &  \\\\ \\hline \n",opcode);
#else
    fprintf(fp,"\\begin{tabular}{|r|r|r|l|r|r|r|r||r|r|} \\hline \n");
    fprintf(fp," & & code & op & loc & loc & loc & loc & double & double \\\\ \\hline \n");
    fprintf(fp," & & %i & start of tape & & & & & & \\\\ \\hline \n",opcode);
#endif
    pagelength = 0;
}

void checkPageBreak() { 
    if (pagelength == 100) { /* 101 lines per page */
        fprintf(fp,"\\end{tabular}\\\\\n");
        fprintf(fp,"\\newpage\n");
#ifdef ADOLC_TAPE_DOC_VALUES
        fprintf(fp,"\\begin{tabular}{|r|r|r|l|r|r|r|r||r|r||r|r|r|r|} \\hline \n");
        fprintf(fp," & & code & op & loc & loc & loc & loc & double & double & value & value & value & value \\\\ \\hline \n");
#else
        fprintf(fp,"\\begin{tabular}{|r|r|r|l|r|r|r|r||r|r|} \\hline \n");
        fprintf(fp," & & code & op & loc & loc & loc & loc & double & double \\\\ \\hline \n");
#endif
        pagelength=-1;
    }
} 

/****************************************************************************/
/* filewrite( opcode number,  op name, number locations, locations, values,           */
/*            number constants, constants )                                 */
/****************************************************************************/
void filewrite( unsigned short opcode, const char* opString, int nloc, int *loc,
                double *val,int ncst, double* cst) {
    int i;

    checkPageBreak();

    /* write opcode counters and  number */
    fprintf(fp,"%i & %i & %i & ",op_cnt, rev_op_cnt, opcode);

    /* write opcode name if available */
    if (opString) fprintf(fp,"%s",opString);
    
    /* write locations (max 4) right-justified */
    fprintf(fp," &");
    if (opcode==ext_diff || opcode==ext_diff_iArr || opcode==ext_diff_v2)
        opcode = ext_diff;
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
#ifdef ADOLC_TAPE_DOC_VALUES /* values + constants */
    /* constants (max 2) */
    if (opcode==ext_diff || opcode == vec_copy)
        nloc=0;
    if (opcode == vec_dot || opcode == vec_axpy)
        nloc=1;
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
        fprintf(fp,"$ %e $",cst[ncst-1]);
    } else {
        fprintf(fp," &");
        fprintf(fp," ");
    }
#endif

    fprintf(fp,"\\\\ \\hline \n"); /* end line */
    fflush(fp);
    pagelength++;
}

#ifdef ADOLC_AMPI_SUPPORT
/****************************************************************************/
/* filewrite_ampi( opcode number,  op name, number locations, locations )   */
/****************************************************************************/
void filewrite_ampi( unsigned short opcode, const char* opString, int nloc, int *loc) {
    int i;

    checkPageBreak();

    /* write opcode counters and  number */
    fprintf(fp,"%i & %i & %i & ",op_cnt, rev_op_cnt, opcode);
    
    /* write opcode name if available */
    if (opString) fprintf(fp,"%s",opString);

#ifdef ADOLC_TAPE_DOC_VALUES /* values + constants */
    fprintf(fp," & \\multicolumn{10}{|l|}{");
#else
    fprintf(fp," & \\multicolumn{6}{|l|}{(");
#endif
    for(i=0; i<(nloc-1); i++) fprintf(fp," %i, ",loc[i]);
    if (nloc) fprintf(fp," %i",loc[nloc-1]);
    fprintf(fp,")} ");
    fprintf(fp,"\\\\ \\hline \n"); /* end line */
    fflush(fp);
    pagelength++;
}
#endif

/*--------------------------------------------------------------------------*/
void filewrite_end( int opcode ) {
#ifdef ADOLC_TAPE_DOC_VALUES
  fprintf(fp," %i & %i & %i & end of tape & & & & & & & & & &  \\\\ \\hline \n",op_cnt,rev_op_cnt, opcode);
#else
    fprintf(fp," %i & %i & %i & end of tape & & & & & & \\\\ \\hline \n",op_cnt,rev_op_cnt,opcode);
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
    int loc_a[maxLocsPerOp];
    double val_a[4]={0,0,0,0}, cst_d[2]={0,0};
    const char* opName;
#ifdef ADOLC_TAPE_DOC_VALUES
	locint qq;
#endif
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /****************************************************************************/
    /*                                                                    INITs */
#ifdef ADOLC_AMPI_SUPPORT
    MPI_Datatype anMPI_Datatype;
    MPI_Comm anMPI_Comm;
    MPI_Request anMPI_Request;
    MPI_Op anMPI_Op;
    int i;
    double aDouble;
#endif
    init_for_sweep(tnum);
    tag = tnum;

    if ((depcheck != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS]) ||
            (indcheck != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS]) ) {
        fprintf(DIAG_OUT,"ADOL-C error: Tape_doc on tape %d  aborted!\n",tag);
        fprintf(DIAG_OUT,"Number of dependent (%d) and/or independent (%d) "
                "variables passed to Tape_doc is\ninconsistent with "
                "number recorded on tape %d (%zu:%zu)\n", depcheck,
                indcheck, tag, ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS],
                ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS]);
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }

    /* globals */
    op_cnt=0;
    rev_op_cnt=ADOLC_CURRENT_TAPE_INFOS.stats[NUM_OPERATIONS]+1;

    dp_T0 = myalloc1(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES]);

    operation=get_op_f();
    ++op_cnt;
    --rev_op_cnt;
    while (operation !=end_of_tape) {
        switch (operation) {

                /****************************************************************************/
                /*                                                                  MARKERS */

                /*--------------------------------------------------------------------------*/
            case end_of_op:                                          /* end_of_op */
  	        filewrite(operation,"end of op",0,loc_a,val_a,0,cst_d);
                get_op_block_f();
                operation=get_op_f();
		++op_cnt;
		--rev_op_cnt;
                /* Skip next operation, it's another end_of_op */
                break;

                /*--------------------------------------------------------------------------*/
            case end_of_int:                                        /* end_of_int */
	        filewrite(operation,"end of int",0,loc_a,val_a,0,cst_d);
                get_loc_block_f();
                break;

                /*--------------------------------------------------------------------------*/
            case end_of_val:                                        /* end_of_val */
	        filewrite(operation,"end of val",0,loc_a,val_a,0,cst_d);
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
                arg  = get_locint_f();
                loc_a[0] = arg;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
#endif
                filewrite(operation,"eq zero",1,loc_a,val_a,0,cst_d);
                break;
            case neq_zero :                                           /* neq_zero */
                arg  = get_locint_f();
                loc_a[0] = arg;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
#endif
                filewrite(operation,"neq zero",1,loc_a,val_a,0,cst_d);
                break;
            case le_zero  :                                            /* le_zero */
                arg  = get_locint_f();
                loc_a[0] = arg;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
#endif
                filewrite(operation,"le zero",1,loc_a,val_a,0,cst_d);
                break;
            case gt_zero  :                                            /* gt_zero */
                arg  = get_locint_f();
                loc_a[0] = arg;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
#endif
                filewrite(operation,"gt zero",1,loc_a,val_a,0,cst_d);
                break;
            case ge_zero  :                                            /* ge_zero */
                arg  = get_locint_f();
                loc_a[0] = arg;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
#endif
                filewrite(operation,"ge zero",1,loc_a,val_a,0,cst_d);
                break;
            case lt_zero  :                                            /* lt_zero */
                arg  = get_locint_f();
                loc_a[0] = arg;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
#endif
                filewrite(operation,"lt zero",1,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]= dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"assign a",2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case assign_d:            /* assign an adouble variable a    assign_d */
                /* double value. (=) */
                res  = get_locint_f();
                cst_d[0]=get_val_f();
                loc_a[0]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                dp_T0[res]= cst_d[0];
                val_a[0]=dp_T0[res];
#endif
                filewrite(operation,"assigne d",1,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case assign_d_one:    /* assign an adouble variable a    assign_d_one */
                /* double value. (1) (=) */
                res  = get_locint_f();
                loc_a[0]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                dp_T0[res]= 1.0;
                val_a[0]=dp_T0[res];
#endif
                filewrite(operation,"assign d one",1,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case assign_d_zero:  /* assign an adouble variable a    assign_d_zero */
                /* double value. (0) (=) */
                res  = get_locint_f();
                loc_a[0]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                dp_T0[res]= 0.0;
                val_a[0]=dp_T0[res];
#endif
                filewrite(operation,"assign d zero",1,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case assign_ind:       /* assign an adouble variable an    assign_ind */
                /* independent double value (<<=) */
                res  = get_locint_f();
                loc_a[0]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                dp_T0[res]= basepoint[indexi];
                cst_d[0]= basepoint[indexi];
                val_a[0]=dp_T0[res];
                filewrite(operation,"assign ind",1,loc_a,val_a,1,cst_d);
#else
                filewrite(operation,"assign ind",1,loc_a,val_a,0,cst_d);
#endif
                indexi++;
                break;

                /*--------------------------------------------------------------------------*/
            case assign_dep:           /* assign a float variable a    assign_dep */
                /* dependent adouble value. (>>=) */
                res = get_locint_f();
                loc_a[0]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[res];
                valuepoint[indexd++]=dp_T0[res];
#endif
                filewrite(operation,"assign dep",1,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                dp_T0[res] += coval;
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,"eq plus d",1,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_plus_a:             /* Add an adouble to another    eq_plus_a */
                /* adouble. (+=) */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]+= dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"eq plus a",2,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res] += dp_T0[arg1]*dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"eq plus prod",3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_d:       /* Subtract a floating point from an    eq_min_d */
                /* adouble. (-=) */
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = res;
                cst_d[0] = coval;
#ifdef ADOLC_TAPE_DOC_VALUES
                dp_T0[res] -= coval;
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,"eq min d",1,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_a:        /* Subtract an adouble from another    eq_min_a */
                /* adouble. (-=) */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]-= dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"eq min a",2,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res] -= dp_T0[arg1]*dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"eq min prod",3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_d:              /* Multiply an adouble by a    eq_mult_d */
                /* flaoting point. (*=) */
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = res;
                cst_d[0] = coval;
#ifdef ADOLC_TAPE_DOC_VALUES
                dp_T0[res] *= coval;
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,"eq mult d",1,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_a:       /* Multiply one adouble by another    eq_mult_a */
                /* (*=) */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]*= dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"eq mult a",2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case incr_a:                        /* Increment an adouble    incr_a */
                res = get_locint_f();
                loc_a[0] = res;
#ifdef ADOLC_TAPE_DOC_VALUES
                dp_T0[res]++;
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,"incr a",1,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case decr_a:                        /* Increment an adouble    decr_a */
                res = get_locint_f();
                loc_a[0] = res;
#ifdef ADOLC_TAPE_DOC_VALUES
                dp_T0[res]--;
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,"decr a",1,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res]=dp_T0[arg1]+dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"plus a a",3,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]= dp_T0[arg] + coval;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"plus d a",2,loc_a,val_a,1,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res]=dp_T0[arg1]-dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"min a a",3,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = coval - dp_T0[arg];
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,"min d a",2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case mult_a_a:               /* Multiply two adoubles (*)    mult_a_a */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res]=dp_T0[arg1]*dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"mult a a",3,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = coval * dp_T0[arg];
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,"mult d a",2,loc_a,val_a,1,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                val_a[1]=dp_T0[arg2];
                dp_T0[res]=dp_T0[arg1]/dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"div a a",3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case div_d_a:             /* Division double - adouble (/)    div_d_a */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = coval / dp_T0[arg];
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,"div d a",2,loc_a,val_a,1,cst_d);
                break;


                /****************************************************************************/
                /*                                                         SIGN  OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case pos_sign_a:                                        /* pos_sign_a */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]= dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"pos sign a",2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case neg_sign_a:                                        /* neg_sign_a */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]= -dp_T0[arg];
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"neg sign a",2,loc_a,val_a,0,cst_d);
                break;


                /****************************************************************************/
                /*                                                         UNARY OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case exp_op:                          /* exponent operation    exp_op */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]= exp(dp_T0[arg]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"exp op",2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case sin_op:                              /* sine operation    sin_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                /* olvo 980923 changed order to allow x=sin(x) */
                val_a[0]=dp_T0[arg1];
                dp_T0[arg2]= cos(dp_T0[arg1]);
                dp_T0[res] = sin(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"sin op",3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case cos_op:                            /* cosine operation    cos_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                /* olvo 980923 changed order to allow x=cos(x) */
                val_a[0]=dp_T0[arg1];
                dp_T0[arg2]= sin(dp_T0[arg1]);
                dp_T0[res] = cos(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"cos op",3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case atan_op:                                              /* atan_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = atan(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"atan op",3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case asin_op:                                              /* asin_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = asin(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"asin op",3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case acos_op:                                              /* acos_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = acos(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"acos op",3,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = asinh(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"asinh op",3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case acosh_op:                                           /* acosh_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = acosh(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"acosh op",3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case atanh_op:                                            /* atanh_op */
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = atanh(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"atanh op",3,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case erf_op:                                                /* erf_op */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                loc_a[2]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = erf(dp_T0[arg1]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"erf op",3,loc_a,val_a,0,cst_d);
                break;

#endif
                /*--------------------------------------------------------------------------*/
            case log_op:                                                /* log_op */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]= log(dp_T0[arg]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"log op",2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case pow_op:                                                /* pow_op */
                arg  = get_locint_f();
                res  = get_locint_f();
                coval   = get_val_f();
                cst_d[0]=coval;
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res] = pow(dp_T0[arg],coval);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"pow op",2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case sqrt_op:                                              /* sqrt_op */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]= sqrt(dp_T0[arg]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"sqrt op",2,loc_a,val_a,0,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case cbrt_op:                                              /* cbrt_op */
                arg  = get_locint_f();
                res  = get_locint_f();
                loc_a[0]=arg;
                loc_a[1]=res;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                dp_T0[res]= cbrt(dp_T0[arg]);
                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                val_a[1]=dp_T0[res];
#endif
                filewrite(operation,"cbrt op",2,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg1];
                dp_T0[res] = cst_d[1];
                val_a[1]=dp_T0[arg2];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"gen quad",3,loc_a,val_a,2,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg1];
                val_a[1] = dp_T0[arg2];
                if (dp_T0[arg1] > dp_T0[arg2])
                    dp_T0[res] = dp_T0[arg2];
                else
                    dp_T0[res] = dp_T0[arg1];
                val_a[2] = dp_T0[res];
#endif
                filewrite(operation,"min op",3,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case abs_val:                                              /* abs_val */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = fabs(dp_T0[arg]);
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,"abs val",2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case ceil_op:                                              /* ceil_op */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = ceil(dp_T0[arg]);
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,"ceil op",2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case floor_op:                 /* Compute ceil of adouble    floor_op */
                arg   = get_locint_f();
                res   = get_locint_f();
                coval = get_val_f();
                loc_a[0] = arg;
                loc_a[1] = res;
                cst_d[0] = coval;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
                dp_T0[res]  = floor(dp_T0[arg]);
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,"floor op",2,loc_a,val_a,1,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                val_a[1]=dp_T0[arg1];
                val_a[2]=dp_T0[arg2];
                if (dp_T0[arg]>0)
                    dp_T0[res]=dp_T0[arg1];
                else
                    dp_T0[res]=dp_T0[arg2];
                val_a[3]=dp_T0[res];
#endif
                filewrite(operation,"cond assign $\\longrightarrow$",4,loc_a,val_a,1,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0]=dp_T0[arg];
                val_a[1]=dp_T0[arg1];
                if (dp_T0[arg]>0)
                    dp_T0[res]=dp_T0[arg1];
                val_a[2]=dp_T0[res];
#endif
                filewrite(operation,"cond assign s $\\longrightarrow$",3,loc_a,val_a,1,cst_d);
                break;

            case vec_copy:
                res = get_locint_f();
                arg = get_locint_f();
                size = get_locint_f();
                loc_a[0] = res;
                loc_a[1] = arg;
                loc_a[2] = size;
#ifdef ADOLC_TAPE_DOC_VALUES
                for(qq=0;qq<size;qq++) 
                    dp_T0[res+qq] = dp_T0[arg+qq];
#endif
                filewrite(operation,"vec copy $\\longrightarrow$",3,loc_a,val_a,0,cst_d);
                break;

            case vec_dot:
                res = get_locint_f();
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                size = get_locint_f();
                loc_a[0] = res;
                loc_a[1] = arg1;
                loc_a[2] = arg2;
                loc_a[3] = size;
#ifdef ADOLC_TAPE_DOC_VALUES
                dp_T0[res] = 0;
                for(qq=0;qq<size;qq++) 
                    dp_T0[res] += dp_T0[arg1+qq] *  dp_T0[arg2+qq];
                val_a[0] = dp_T0[res];
#endif
                filewrite(operation,"vec dot $\\longrightarrow$",4,loc_a,val_a,0,cst_d);
                break;

            case vec_axpy:
                res = get_locint_f();
                arg = get_locint_f();
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                size = get_locint_f();
                loc_a[0] = res;
                loc_a[1] = arg;
                loc_a[1] = arg1;
                loc_a[2] = arg2;
                loc_a[3] = size;
#ifdef ADOLC_TAPE_DOC_VALUES
                val_a[0] = dp_T0[arg];
                for(qq=0;qq<size;qq++) 
                    dp_T0[res+qq] = dp_T0[arg] * dp_T0[arg1+qq] + dp_T0[arg2+qq];
#endif
                filewrite(operation,"vec axpy $\\longrightarrow$",4,loc_a,val_a,0,cst_d);
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
#ifdef ADOLC_TAPE_DOC_VALUES
                for (l=0; l<size; l++)
                    dp_T0[res+l] = d[l];
                val_a[0] = make_nan();
                val_a[1] = dp_T0[res];
#endif
                filewrite(operation,"take stock op",2,loc_a,val_a,1,cst_d);
                break;

                /*--------------------------------------------------------------------------*/
            case death_not:                                          /* death_not */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                loc_a[0]=arg1;
                loc_a[1]=arg2;
                filewrite(operation,"death not",2,loc_a,val_a,0,cst_d);
                break;

                /****************************************************************************/
            case ext_diff:
                loc_a[0] = get_locint_f() + 1; /* index */
                loc_a[1] = get_locint_f(); /* n */
                loc_a[2] = get_locint_f(); /* m */
                loc_a[3] = get_locint_f(); /* xa[0].loc */
                loc_a[3] = get_locint_f(); /* ya[0].loc */
                loc_a[3] = get_locint_f(); /* dummy */
                filewrite(operation, "extern diff",3, loc_a, val_a, 0, cst_d);
                break;

            case ext_diff_iArr:
                loc_a[0] = get_locint_f(); /* iArr length */
                for (l=0; l<loc_a[0];++l) get_locint_f(); /* iArr */
                get_locint_f(); /* iArr length again */
                loc_a[0] = get_locint_f() + 1; /* index */
                loc_a[1] = get_locint_f(); /* n */
                loc_a[2] = get_locint_f(); /* m */
                loc_a[3] = get_locint_f(); /* xa[0].loc */
                loc_a[3] = get_locint_f(); /* ya[0].loc */
                loc_a[3] = get_locint_f(); /* dummy */
                filewrite(operation, "extern diff iArr",3, loc_a, val_a, 0, cst_d);
                break;
            case ext_diff_v2:
                loc_a[0] = get_locint_f(); /* index */
                loc_a[1] = get_locint_f(); /* iArr length */
                for (l=0; l<loc_a[1];++l) get_locint_f(); /* iArr */
                get_locint_f(); /* iArr length again */
                loc_a[1] = get_locint_f(); /* nin */
                loc_a[2] = get_locint_f(); /* nout */
                for (l=0; l<loc_a[1];++l) { get_locint_f(); get_locint_f(); } 
                /* input vectors sizes and start locs */
                for (l=0; l<loc_a[2];++l) { get_locint_f(); get_locint_f(); } 
                /* output vectors sizes and start locs */
                get_locint_f(); /* nin again */
                get_locint_f(); /* nout again */
                filewrite(operation, "extern diff v2",3, loc_a, val_a, 0, cst_d);
                break;
#ifdef ADOLC_MEDIPACK_SUPPORT
                /*--------------------------------------------------------------------------*/
            case medi_call:
                loc_a[0] = get_locint_f();

                /* currently not supported */
                break;
#endif
#ifdef ADOLC_AMPI_SUPPORT
            case ampi_send:
	        loc_a[0] = get_locint_f();   /* start loc */
	        TAPE_AMPI_read_int(loc_a+1); /* count */
	        TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
	        TAPE_AMPI_read_int(loc_a+2); /* endpoint */
	        TAPE_AMPI_read_int(loc_a+3); /* tag */
	        TAPE_AMPI_read_int(loc_a+4); /* pairedWith */
	        TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
		filewrite_ampi(operation, "ampi send",5, loc_a);
		break; 

            case ampi_recv:
                loc_a[0] = get_locint_f();   /* start loc */
                TAPE_AMPI_read_int(loc_a+1); /* count */
                TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
                TAPE_AMPI_read_int(loc_a+2); /* endpoint */
                TAPE_AMPI_read_int(loc_a+3); /* tag */
                TAPE_AMPI_read_int(loc_a+4); /* pairedWith */
                TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
                filewrite_ampi(operation, "ampi recv",5, loc_a);
                break;

            case ampi_isend: 
              /* push is delayed to the accompanying completion */
              TAPE_AMPI_read_MPI_Request(&anMPI_Request);
              filewrite_ampi(operation, "ampi isend",0, loc_a);
              break;

            case ampi_irecv:
              /* push is delayed to the accompanying completion */
              TAPE_AMPI_read_MPI_Request(&anMPI_Request);
              filewrite_ampi(operation, "ampi irecv",0, loc_a);
              break;

            case ampi_wait: 
	      /* for the operation we had been waiting for */
              size=0;
              loc_a[size++] = get_locint_f(); /* start loc */
              TAPE_AMPI_read_int(loc_a+size++); /* count */
              TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
              TAPE_AMPI_read_int(loc_a+size++); /* endpoint */
              TAPE_AMPI_read_int(loc_a+size++); /* tag */
              TAPE_AMPI_read_int(loc_a+size++); /* pairedWith */
              TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
              TAPE_AMPI_read_MPI_Request(&anMPI_Request);
              TAPE_AMPI_read_int(loc_a+size++); /* origin */
              filewrite_ampi(operation, "ampi wait",size, loc_a);
              break;

            case ampi_barrier:
              TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
              filewrite_ampi(operation, "ampi barrier",0, loc_a);
              break;

	    case ampi_bcast:
	      loc_a[0] = get_locint_f();   /* start loc */
	      TAPE_AMPI_read_int(loc_a+1); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
	      TAPE_AMPI_read_int(loc_a+2); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      filewrite_ampi(operation, "ampi bcast",3, loc_a);
	      break;

	    case ampi_reduce:
	      loc_a[0] = get_locint_f();   /* rbuf */
	      loc_a[1] = get_locint_f();   /* sbuf */
	      TAPE_AMPI_read_int(loc_a+2); /* count */
	      TAPE_AMPI_read_int(loc_a+3); /* pushResultData */
	      i=0; /* read stored double array into dummy variable */
	      while (i<loc_a[2]) { TAPE_AMPI_read_double(&aDouble); i++; }
	      if (loc_a[3]) {
	        i=0; /* for root, also read stored reduction result */
	        while (i<loc_a[2]) { TAPE_AMPI_read_double(&aDouble); i++; }
	      }
	      TAPE_AMPI_read_int(loc_a+3); /* pushResultData again */
	      TAPE_AMPI_read_MPI_Op(&anMPI_Op);
	      TAPE_AMPI_read_int(loc_a+4); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
	      TAPE_AMPI_read_int(loc_a+2); /* count again */
	      filewrite_ampi(operation, "ampi reduce",5, loc_a);
	      break;

	    case ampi_allreduce:
	      loc_a[0] = get_locint_f();   /* rbuf */
	      loc_a[1] = get_locint_f();   /* sbuf */
	      TAPE_AMPI_read_int(loc_a+2); /* count */
	      TAPE_AMPI_read_int(loc_a+3); /* pushResultData */
	      i=0; /* read off stored double array into dummy variable */
	      while (i<loc_a[2]) { TAPE_AMPI_read_double(&aDouble); i++; }
	      if (loc_a[3]) {
	        i=0; /* for root, also read off stored reduction result */
	        while (i<loc_a[2]) { TAPE_AMPI_read_double(&aDouble); i++; }
	      }
	      TAPE_AMPI_read_int(loc_a+3); /* pushResultData again */
	      TAPE_AMPI_read_MPI_Op(&anMPI_Op);
	      TAPE_AMPI_read_int(loc_a+4); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
	      TAPE_AMPI_read_int(loc_a+2); /* count again */
	      filewrite_ampi(operation, "ampi allreduce",5, loc_a);
	      break;

	    case ampi_gather:
	      size=0;
	      TAPE_AMPI_read_int(loc_a+size++); /* commSizeForRootOrNull */
	      if (*(loc_a+0)>0) {
	        loc_a[size++] = get_locint_f(); /* rbuf loc */
	        TAPE_AMPI_read_int(loc_a+size++); /* rcnt */
	        TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
	      }
	      loc_a[size++]=get_locint_f(); /* buf loc */
	      TAPE_AMPI_read_int(loc_a+size++); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
	      TAPE_AMPI_read_int(loc_a+size++); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_int(loc_a+0); /* commSizeForRootOrNull */
	      filewrite_ampi(operation, "ampi gather",size, loc_a);
	      break;

	    case ampi_scatter:
	      size=0;
	      TAPE_AMPI_read_int(loc_a+size++); /* commSizeForRootOrNull */
	      if (*(loc_a+0)>0) {
	        loc_a[size++] = get_locint_f(); /* rbuf loc */
	        TAPE_AMPI_read_int(loc_a+size++); /* rcnt */
	        TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
	      }
	      loc_a[size++]=get_locint_f(); /* buf loc */
	      TAPE_AMPI_read_int(loc_a+size++); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
	      TAPE_AMPI_read_int(loc_a+size++); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_int(loc_a+0); /* commSizeForRootOrNull */
	      filewrite_ampi(operation, "ampi scatter",size, loc_a);
	      break;

	    case ampi_allgather:
	      TAPE_AMPI_read_int(loc_a+1); /* commSizeForRootOrNull */
	      if (*(loc_a+1)>0) {
	        TAPE_AMPI_read_int(loc_a+2); /* rcnt */
	        loc_a[2] = get_locint_f(); /* rbuf loc */
	        TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
	      }
	      TAPE_AMPI_read_int(loc_a+3); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_int(loc_a+1); /* commSizeForRootOrNull */
	      filewrite_ampi(operation, "ampi allgather",4, loc_a);
	      break;

	    case ampi_gatherv:
	      size=0;
	      TAPE_AMPI_read_int(loc_a+size++); /* commSizeForRootOrNull */
	      if (*(loc_a+0)>0) {
	        loc_a[size++] = get_locint_f(); /* rbuf loc */
	        TAPE_AMPI_read_int(loc_a+size++); /* rcnt[0] */
	        TAPE_AMPI_read_int(loc_a+size++); /* displs[0] */
	      }
	      for (l=1;l<*(loc_a+0);++l) {
	        TAPE_AMPI_read_int(loc_a+size);
	        TAPE_AMPI_read_int(loc_a+size);
	      }
	      if (*(loc_a+0)>0) {
	        TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
	      }
              loc_a[size++] = get_locint_f(); /* buf loc */
	      TAPE_AMPI_read_int(loc_a+size++); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
	      TAPE_AMPI_read_int(loc_a+size++); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_int(loc_a+0); /* commSizeForRootOrNull */
	      filewrite_ampi(operation, "ampi gatherv",size, loc_a);
		break;

            case ampi_scatterv: 
              size=0;
              TAPE_AMPI_read_int(loc_a+size++); /* commSizeForRootOrNull */
              if (*(loc_a+0)>0) {
                loc_a[size++] = get_locint_f(); /* rbuf loc */
                TAPE_AMPI_read_int(loc_a+size++); /* rcnt[0] */
                TAPE_AMPI_read_int(loc_a+size++); /* displs[0] */
              }
              for (l=1;l<*(loc_a+0);++l) {
                TAPE_AMPI_read_int(loc_a+size);
                TAPE_AMPI_read_int(loc_a+size);
              }
              if (*(loc_a+0)>0) {
                TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
              }
              loc_a[size++] = get_locint_f(); /* buf loc */
              TAPE_AMPI_read_int(loc_a+size++); /* count */
              TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
              TAPE_AMPI_read_int(loc_a+size++); /* root */
              TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
              TAPE_AMPI_read_int(loc_a+0); /* commSizeForRootOrNull */
              filewrite_ampi(operation, "ampi scatterv",size, loc_a);
              break;

            case ampi_allgatherv:
	      size=0;
	      TAPE_AMPI_read_int(loc_a+size++); /* commSizeForRootOrNull */
	      for (l=0;l<*(loc_a);++l) {
		TAPE_AMPI_read_int(loc_a+size); /* rcnts */
		TAPE_AMPI_read_int(loc_a+size+1); /* displs */
	      }
	      if (*(loc_a)>0) {
		size+=2;
		loc_a[size++] = get_locint_f(); /* rbuf loc */
		TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
	      }
              loc_a[size++] = get_locint_f(); /* buf loc */
	      TAPE_AMPI_read_int(loc_a+size++); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
	      TAPE_AMPI_read_int(loc_a+size++); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_int(loc_a); /* commSizeForRootOrNull */
	      filewrite_ampi(operation, "ampi allgatherv",size, loc_a);
	      break;
#endif
                /*--------------------------------------------------------------------------*/
            default:                                                   /* default */
                /* Die here, we screwed up */
                fprintf(DIAG_OUT,"ADOL-C error: Fatal error in tape_doc for op %d\n",
                        operation);
                break;

        } /* endswitch */

        /* Read the next operation */
        operation=get_op_f();
	++op_cnt;
	--rev_op_cnt;
    }  /* endwhile */

    if (operation == end_of_tape) {
        filewrite_end(operation);
    };

    if (dp_T0) free(dp_T0);
    dp_T0 = NULL;

    end_sweep();
} /* end tape_doc */


/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS
