/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     checkpointing_p.h
 Revision: $Id$
 Contents: private functions for the checkpointing functions.
 
 Copyright (c) Andreas Kowarz, Jean Utke
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
   
----------------------------------------------------------------------------*/

#if !defined(ADOLC_CHECKPOINTING_P_H)
#define ADOLC_CHECKPOINTING_P_H 1

#include <adolc/internal/common.h>
#include <adolc/checkpointing.h>
#include "buffer_temp.h"
#include "taping_p.h"

#include <stack>
using std::stack;

BEGIN_C_DECLS
/****************************************************************************/
/*                                                         Now the C THINGS */

#define CP_BLOCK_SIZE 10

CpInfos *get_cp_fct(int index);

void init_CpInfos(CpInfos *cpInfos);

END_C_DECLS

/****************************************************************************/

#if defined(__cplusplus)

#define ADOLC_BUFFER_TYPE Buffer< CpInfos, CP_BLOCK_SIZE >
extern ADOLC_BUFFER_TYPE ADOLC_EXT_DIFF_FCTS_BUFFER_DECL;

/* field of pointers to the value fields of a checkpoint */
typedef double **StackElement;
extern stack<StackElement> ADOLC_CHECKPOINTS_STACK_DECL;

/* a cleanup function */
void cp_clearStack();

#endif

#endif /* ADOLC_CHECKPOITING_P_H */

