#ifndef LINALG_H
#define LINALG_H

#include "nr3.h"

namespace scilib{

    struct LUdcmp {
        Int n;
        MatDoub lu; // stores decomposition
        VecInt indx; // stores permutation
        Doub d; // used by det
        LUdcmp(MatDoub_I &a); // constructor for decomposition
        void solve(VecDoub_I &b, VecDoub_O &x); // solve single right-hand side
        void solve(Matdoub_I &b, MatDoub_O &x); // solve multiple right-hand sides
        void inverse(MatDoub_O &ainv); // inverse of matrix
        Doub det(); // determinate
    }
}


#endif // LINALG_H