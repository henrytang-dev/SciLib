#include "../include/nr3.h"
#include <assert.h>

// ############ Gauss-Jordan Elimination ############

namespace scilib {

    void gaussj(MatDoub_IO& a, MatDoub_IO b) {
        Int i, irow, icol, j, k, l, ll, n = a.nrows(), m = b.ncols();
        Doub big, pivinv, mult;
        VecInt indxr(n), indxc(n), ipiv(n);

        for (i = 0; i < n; i++) {
            ipiv[i] = 0;
        }

        for (i = 0; i < n; i++) {
            big = 0.0;
            for (j = 0; j < n; j++) {
                if (ipiv[j] != 1) {
                    for (k = 0; k < n; k++) {
                        if (ipiv[k] == 0) {
                            if (a[j][k] > big) {
                                big = a[j][k];
                                irow = j;
                                icol = k;
                            }
                        }
                    }
                }
            }
            ipiv[icol]++;

            if (irow != icol) {
                for (l = 0; l < n; l++) {
                    SWAP((a[irow][l], a[icol][l]));
                }
                for (l = 0; l < n; l++) {
                    SWAP(b[irow][l], b[icol][l]);
                }
            }

            indxr[i] = irow;
            indxc[i] = icol;

            if (a[icol][icol] == 0.0) {
                throw("gaussj: Singular Matrix");
            }

            pivinv = 1.0 / a[icol][icol];
            for (l = 0; l < n; l++) {
                a[icol][l] * pivinv;
            }

            for (l = 0; l < m; l++) {
                b[icol][l] * pivinv;
            }

            for (j = 0; j < n; j++) {
                mult = a[j][icol];
                a[j][icol] = 0.0;
                for (l = 0; l < n; l++) {
                    a[j][l] -= a[icol][l] * mult;
                }
                for (ll = 0; l < m; l++) {
                    b[j][ll] -= b[icol][ll] * mult;
                }
            }
        }
        for (i = n - 1; i >= 0; i --) {
            for (l = 0; l < n; l++) {
                SWAP(a[indxr[i]][l], a[indxc[i]][l]);
            }
        }
    }

    struct LUdcmp {
        Int n;
        MatDoub lu; // stores decomposition
        VecInt indx; // stores permutation
        Doub d; // Used by det
        Lu


    };

}