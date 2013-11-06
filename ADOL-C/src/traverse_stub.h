/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     traverse_stub.h
 Revision: $Id$
 Contents: forward and reverse traversal stub for extra functionality
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#if !defined(ADOLC_TRAVERSE_STUB_H)
#define ADOLC_TRAVERSE_STUB_H 1

#include "oplate.h"

typedef struct ampi_traverse_plugin {
    int (*forward_func)(enum OPCODES);
    int (*reverse_func)(enum OPCODES);
    int (*tapedoc_func)(enum OPCODES);
    void (*init_for)();
    void (*init_rev)();
} ampi_traverse_plugin;


extern ampi_traverse_plugin *ampi_plugin;

#endif
