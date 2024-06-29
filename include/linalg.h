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
        void solve(MatDoub_I &b, MatDoub_O &x); // solve multiple right-hand sides
        void inverse(MatDoub_O &ainv); // inverse of matrix
        Doub det(); // determinate
        void mprove(VecDoub_I &b, VecDoub_IO &x);
        MatDoub_I& aref;
    };

    struct SVD {
        Int m, n;
        MatDoub u, v;
        VecDoub w;
        Doub eps, tsh;
        SVD(MatDoub_I &a) : m(a.nrows()), n(a.ncols()), u(a), v(n, n), w(n) {
            eps = numeric_limits<Doub>::epsilon();
            decompose();
            reorder();
            tsh = 0.5 * sqrt(m + n + 1.) * w[0] * eps;
        }

        void solve(VecDoub_I &b, VecDoub_O &x, Doub thresh);
        void solve(MatDoub_I &b, MatDoub_O &x, Doub thresh);

        Int rank(Doub thresh);
        Int nullity(Doub thresh);
        MatDoub range(Doub thresh);
        MatDoub nullspace(Doub thresh);

        Doub inv_condition() {
                return (w[0] <= 0. || w[n - 1] <= 0.) ? 0. : w[n - 1] / w[0];
        }

        void decompose();
        void reorder();
        Doub pythag(const Doub a, const Doub b);
    };

    struct QRdcmp {
        Int n;
        MatDoub qt, r;
        Bool sing;
        QRdcmp(MatDoub_I& a);
        void solve(VecDoub_I& b, VecDoub_O& x);
        void qtmult(VecDoub_I &b, VecDoub_O &x);
        void rsolve(VecDoub_I &b, VecDoub_O &x);
        void update(VecDoub_I &u, VecDoub_I &v);
        void rotate(const Int i, const Doub a, const Doub b);
    };
}


#endif // LINALG_H