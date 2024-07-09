#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include "lstm.h"

void XavierInit(Eigen::MatrixXf& W);
Eigen::MatrixXf clipGrad(const Eigen::MatrixXf& gradients, float clip_val);
void* aligned_alloc(size_t size, size_t alignment);
void free_alloc(void *ptr);
float computeLoss(const std::vector<Eigen::MatrixXf> predictions, const std::vector<Eigen::MatrixXf> targets);
void trainModel(LSTMCell& lstm, const std::vector<Eigen::MatrixXf>& inputs, const std::vector<Eigen::MatrixXf>& targets, int epochs, float learningRate);


#endif // UTILS_H