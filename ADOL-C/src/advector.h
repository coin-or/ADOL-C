/* ---------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++

 Revision: $Id$
 Contents: advector.h contains a vector<adouble> implementation
           that is able to trace subscripting operations.

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#if !defined(ADOLC_ADVECTOR_H)
#define ADOLC_ADVECTOR_H 1

/****************************************************************************/
/*                                                         THIS FILE IS C++ */
#ifdef __cplusplus
#include <vector>

#include <adolc/adouble.h>

/****************************************************************************/
/*                                           THIS IS ONLY FOR TAPED VERSION */
#if !defined(TAPELESS)

class ADOLC_DLL_EXPORT advector {
private:
    std::vector<adouble> data;
    bool nondecreasing() const;
public:
    advector() : data() {}
    explicit advector(size_t n) : data(n) {}
    ~advector() { data.~vector<adouble>(); }
    operator const vector<adouble>&() const { return data; }
    operator vector<adouble>&() { return data; }
    const adouble& operator[](const badouble& index) const;
    adouble& operator[](const badouble& index);
    adouble lookupindex(const badouble& x, const badouble& y) const;
};

#endif /* TAPELESS */
#endif /* __cplusplus */
#endif /* ADOLC_ADVECTOR_H */
