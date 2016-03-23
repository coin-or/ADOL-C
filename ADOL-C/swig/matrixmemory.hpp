/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     matrixmemory.hpp
 Revision: $Id$
 Contents: C allocation of arrays of doubles in several dimensions 

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

--------------------------------------------------------------------------*/
#ifndef MATRIXMEMORY_HPP
#define MATRIXMEMORY_HPP

/*--------------------------------------------------------------------------*/
template <typename Type>
static inline char* populate_dpp_with_contigdata(Type ***const pointer, char *const memory,
                                   int n, int m, Type *const data) {
    char* tmp;
    Type **tmp1; Type *tmp2;
    int i,j;
    tmp = (char*)memory;
    tmp1 = (Type**) memory;
    *pointer = tmp1;
    tmp = (char*)(tmp1+n);
    tmp2 = data;
    for (i=0;i<n;i++) {
        (*pointer)[i] = tmp2;
        tmp2 += m;
    }
    return tmp;
}
/*--------------------------------------------------------------------------*/
template <typename Type>
static inline char* populate_dppp_with_contigdata(Type ****const pointer, char *const memory, 
                                    int n, int m, int p, Type *const data) {
    char* tmp;
    Type ***tmp1; Type **tmp2; Type *tmp3;
    int i,j;
    tmp = (char*) memory;
    tmp1 = (Type***) memory;
    *pointer = tmp1;
    tmp = (char*)(tmp1+n);
    tmp2 = (Type**)tmp;
    for(i=0; i<n; i++) {
        (*pointer)[i] = tmp2;
        tmp2 += m;
    }
    tmp = (char*)tmp2;
    tmp3 = data;
    for(i=0;i<n;i++)
        for(j=0;j<m;j++) {
            (*pointer)[i][j] = tmp3;
            tmp3 += p;
        }
    return tmp;
}
/*--------------------------------------------------------------------------*/
#endif
