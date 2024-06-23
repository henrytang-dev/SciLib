#include "../include/nr3.h"
#include "../include/linalg.h"

Int scilib::SVD::rank(Doub thresh = -1.) {
    Int j, nr=0;
    tsh = (thresh >= 0. ? thresh : 0.5*sqrt(m + n + 1)*w[0]*eps);
    for (j = 0; j < n; j++) {
        if (w[j] > tsh) {
            nr++;
        }
    }
    return nr;
}

Int scilib::SVD::nullity(Doub thresh = -1.) {
    Int j, nn = 0;
    tsh = (thresh >= 0. ? thresh : 0.5*sqrt(m + n + 1.)*w[0] *eps);
    for (j = 0; j < n; j++) {
        if (w[j] <= tsh) {
            nn++;
        }
    }
    return nn;
}

MatDoub scilib::SVD::range(Doub thresh = -1.) {
    Int i, j, nr=0;
    MatDoub range(m, rank(thresh));
    for (j = 0; j < n; j++) {
        if (w[j] > tsh) {
            for (i = 0; i < m; i++) {
                range[i][nr] = u[i][j];
            }
            nr++;
        }
    }
    return range;
}

MatDoub scilib::SVD::nullspace(Doub thresh = -1.) {
    Int j, jj, nn = 0;
    MatDoub nullsp(n, nullity(thresh));
    for (j = 0; j < n; j++) {
        if (w[j] <= tsh) {
            for (jj = 0; jj < n; jj++) {
                nullsp[jj][m] = v[jj][j];
            }
            nn++;
        }
    }
    return nullsp;
}