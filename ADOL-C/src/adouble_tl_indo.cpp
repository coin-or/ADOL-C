/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adouble_tl_indo.cpp
 Revision: $Id$
 Contents: adouble_tl.cpp contains that definitions of procedures used for
sparse patterns.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adtl_indo.h>
#include <adolc/dvlparms.h>
#include <iostream>
#include <vector>

using std::cout;

namespace ADOLC::Sparse::adtl_indo {

/*******************  i/o operations  ***************************************/
ostream &operator<<(ostream &out, const adouble &a) {
  out << a.val;
  return out;
}

istream &operator>>(istream &in, adouble &a) {
  in >> a.val;
  return in;
}

/**************** ADOLC_TRACELESS_SPARSE_PATTERN ****************************/
int ADOLC_Init_sparse_pattern(adouble *a, int n, uint start_cnt) {
  for (unsigned int i = 0; i < n; i++) {
    a[i].delete_pattern();
    a[i].pattern.push_back(i + start_cnt);
  }
  return 3;
}

int ADOLC_get_sparse_pattern(const adouble *b, int m,
                             std::vector<uint *> &pat) {
  pat = std::vector<uint *>(m);
  for (int i = 0; i < m; i++) {
    if (b[i].get_pattern_size() > 0) {
      pat[i] = new uint[b[i].get_pattern_size() + 1];
      pat[i][0] = b[i].get_pattern_size();
      const list<unsigned int> &tmp_set = b[i].get_pattern();
      unsigned int l = 1;
      for (auto it = tmp_set.begin(); it != tmp_set.end(); it++, l++)
        pat[i][l] = *it;
    } else {
      pat[i] = new uint[1];
      pat[i][0] = 0;
    }
  }
  return 3;
}

} // namespace ADOLC::Sparse::adtl_indo
