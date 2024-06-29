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

void scilib::SVD::solve(VecDoub_I &b, VecDoub_O &x, Doub thresh = -1.) {
    Int i, j, jj;
    Doub s;
    if (b.size() != m || x.size()  != n) {
        throw ("SVD: Solve bad sizes");
    }
    VecDoub tmp(n);
    tsh =  (thresh >= 0. ? thresh : 0.5*sqrt(m+n+1.)*w[0]*eps);
    for (j = 0; j < n; j++) {
        s=0.0;
        if (w[j] > tsh) {
            for (i = 0; i < m; i++) {
                s += u[i][j] * b[i];
                s /= w[j];
            }
        }
        tmp[j] = s;
    }
    for (j = 0; j < n; j++) {
        s=0.0;
        for (jj = 0; jj < n; jj++) {
            s += v[j][jj] * tmp[jj];
            x[j] = s;
        }
    }
}

void scilib::SVD::solve(MatDoub_I& b, MatDoub_O& x, Doub thresh = -1.) {
    Int i, j, m = b.ncols();
    if (b.nrows() != n || b.ncols() != m || b.ncols() != x.ncols()) {
        throw ("SVD: Solve bad shapes");
    }
    VecDoub xx(n);
    for (j = 0; j < m; j++) {
        for (i = 0; i < n; i++) {
            xx[i] = b[i][j];
        }
        solve(xx, xx, thresh);
        for (i = 0; i < n; i++) {
            x[i][j] = xx[j];
        }
    }
}