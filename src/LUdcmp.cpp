#include "../include/linalg.h"
#include <assert.h>
#include "../include/nr3.h"

// ############ LU Decomposition ############

scilib::LUdcmp::LUdcmp(MatDoub_I &a) : n(a.nrows()), lu(a), aref(a), indx(n) {
    const Doub TINY = 1.0e-40;
    Int i, j, k, imax;
    Doub big, temp;

    VecDoub vv(n); // implicit scaling of each row
    d = 1.0; // no row interchanges yet

    // Compute scaling factors
    for (i = 0; i < n; i++) {
        big = 0.0;
        for (j = 0; j < n; j++) {
            if ((temp = abs(lu[i][j])) > big) {
                big = temp;
            }
        }
        if (big == 0.0) {
            throw runtime_error("Cannot perform LU decomposition on singular matrix");
        }
        vv[i] = 1.0 / big;
    }

    // Main decomposition loop
    for (k = 0; k < n; k++) {
        big = 0.0; // to store largest element (pivot)
        for (i = k; i < n; i++) {
            temp = vv[i] * abs(lu[i][k]);
            if (temp > big) {
                big = temp;
                imax = i;
            }
        }

        // Pivoting
        if (k != imax) {
            for (j = 0; j < n; j++) {
                temp = lu[imax][j];
                lu[imax][j] = lu[k][j];
                lu[k][j] = temp;
            }
            d = -d; // Adjust the sign of the determinant
            vv[imax] = vv[k];
        }
        indx[k] = imax;

        if (lu[k][k] == 0.0) {
            lu[k][k] = TINY; // Prevent division by zero
        }

        // Elimination
        for (i = k + 1; i < n; i++) {
            temp = lu[i][k] /= lu[k][k]; // Normalize the current element in L
            for (j = k + 1; j < n; j++) {
                lu[i][j] -= temp * lu[k][j]; // Update the remaining elements in U
            }
        }
    }
}

void scilib::LUdcmp::solve(VecDoub_I &b, VecDoub_O &x) {
    Int i, ii = 0, ip, j;
    Doub sum;
    if (b.size() != n || x.size() != n) {
        throw ("LUdcmp::bad sizes error");
    }

    for (i = 0; i < n; i++) {
        x[i] = b[i];
    }
    for (i = 0; i < n; i++) {
        ip = indx[i];
        sum = x[ip];
        x[ip] = x[i];
        if (ii != 0) {
            for (j = ii - 1; j < i; j++) {
                sum -= lu[i][j]*x[j];
            }
        } else if (sum != 0.0) {
            ii = i + 1;
        }
        x[i] = sum;
    }
    for (i = n - 1; i >= 0; i--) {
        sum = x[i];
        for (j = i + 1; j < n; j++) {
            sum -= lu[i][j]*x[j];
        }
        x[i] = sum / lu[i][i];
    }
}

// b: n x m
void scilib::LUdcmp::solve(MatDoub_I &b, MatDoub_O &x) {
    int i, j, m=b.ncols();
    if (b.nrows() != n || x.nrows() != n || b.ncols() != x.ncols()) {
        throw ("LUDcmp::Bad sizes");
    }
    VecDoub xx(n);
    for (j = 0; j < m; j++) {
        for (i = 0; i < n; i++) {
            xx[i] = b[i][j];
        }
        solve(xx, xx);
        for (i = 0; i < n; i++) {
            x[i][j] = xx[i];
        }
    }
}

void scilib::LUdcmp::inverse(MatDoub_O &ainv) {
    Int i, j;
    ainv.resize(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            ainv[i][j] = 1.0;
        }
        solve(ainv, ainv);
    }
}

Doub scilib::LUdcmp::det() {
    Doub dd = d;
    for (int i = 0; i < n; i++) {
        dd *= lu[i][i];
    }
    return dd;
}

void scilib::LUdcmp::mprove(VecDoub_I &b, VecDoub_IO &x) {
    Int i, j;
    VecDoub r(n);
    for (i = 0; i < n; i++) {
        Ldoub sdp = -b[ i];
        for (j = 0; j < n; j++) {
            sdp += (Ldoub) aref[i][j] * (Ldoub)x[j];
        }
        r[i] = sdp;
    }
    solve(r, r);
    for (i = 0; i < n; i++) {
        x[i] -= r[i];
    }
}