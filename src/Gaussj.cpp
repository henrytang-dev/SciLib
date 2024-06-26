#include "../include/nr3.h"
#include <assert.h>

// ############ Gauss-Jordan Elimination ############


void gaussj(MatDoub_IO& a, MatDoub_IO& b) {
    Int i, irow = 0, icol = 0, j, k, l, ll, n = a.nrows(), m = b.nrows();
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
                        if (abs(a[j][k]) > big) {
                            big = abs(a[j][k]);
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
                SWAP(a[irow][l], a[icol][l]);
            }
            for (l = 0; l < m; l++) {
                SWAP(b[irow][l], b[icol][l]);
            }
        }

        indxr[i] = irow;
        indxc[i] = icol;

        if (a[icol][icol] == 0.0) {
            throw runtime_error("gaussj: Singular Matrix");
        }

        pivinv = 1.0 / a[icol][icol];
        a[icol][icol] = 1.0; // Set the pivot to 1
        for (l = 0; l < n; l++) {
            a[icol][l] *= pivinv;
        }

        for (l = 0; l < m; l++) {
            b[icol][l] *= pivinv;
        }

        for (j = 0; j < n; j++) {
            if (j != icol) {
                mult = a[j][icol];
                a[j][icol] = 0.0;
                for (l = 0; l < n; l++) {
                    a[j][l] -= a[icol][l] * mult;
                }
                for (ll = 0; ll < m; ll++) {
                    b[j][ll] -= b[icol][ll] * mult;
                }
            }
        }
    }
    for (i = n - 1; i >= 0; i--) {
        if (indxr[i] != indxc[i]) {
            for (l = 0; l < n; l++) {
                SWAP(a[l][indxr[i]], a[l][indxc[i]]);
            }
            for (l = 0; l < m; l++) {
                SWAP(b[l][indxr[i]], b[l][indxc[i]]);
            }
        }
    }
}

