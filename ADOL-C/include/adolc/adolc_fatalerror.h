/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc_fatalerror.h
 Revision: $Id$
 Contents: handling of fatal errors

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

#ifndef ADOLC_FATALERROR_H
#define ADOLC_FATALERROR_H

#ifdef __cplusplus

#ifndef SWIG
#include <exception>
#include <cstdio>
#endif

class FatalError : public std::exception {
protected:
    static const int MAX_MSG_SIZE = 4*1024;
    char msg[MAX_MSG_SIZE];

public:
    explicit FatalError(int errorcode, const char* what, const char* function, const char* file, int line) {
        // need to use C-style functions that do not use exceptions themselves
        snprintf(this->msg, MAX_MSG_SIZE, "errorcode=%d function=%s file=%s line=%d what=%s", errorcode, function, file, line, what);
    }

    virtual const char* what() const throw() {
        return msg;
    }
};

#endif
#endif
