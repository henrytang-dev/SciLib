#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <Eigen/Dense>
#include <iostream>
#include <cmath>

// UTILS

bool matricesApproxEqual(const Eigen::MatrixXf& mat1, const Eigen::MatrixXf& mat2, float tol = 1e-5) {
    return (mat1 - mat2).array().abs().maxCoeff() < tol;
}

void printTestResult(const std::string& testName, bool result) {
    if (result) {
        std::cout << testName << " passed." << std::endl;
    } else {
        std::cout << testName << " failed." << std::endl;
    }
}


#endif // TEST_UTILS_H