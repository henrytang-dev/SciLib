#include "../include/utils.h"
#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>

void XavierInit(Eigen::MatrixXf& W) {
    double scale = std::sqrt(6.0 / (W.rows() + W.cols()));
    W = Eigen::MatrixXf::Random(W.rows(), W.cols()) * scale;
}

Eigen::MatrixXf clipGrad(const Eigen::MatrixXf& grads, float clip_value) {
    return grads.unaryExpr([clip_value](float x) { return std::min(std::max(-clip_value, x), clip_value); });
}

// Custom aligned allocator
void *aligned_alloc(size_t size, size_t alignment) {
    void *ptr = _aligned_malloc(size, alignment);
    if (!ptr) {
        throw std::bad_alloc();
    }

    return ptr;
}

void free_alloc(void *ptr) {
    _aligned_free(ptr);
}

float computeLoss(const std::vector<Eigen::MatrixXf> predictions, const std::vector<Eigen::MatrixXf> targets) {
    double loss = 0.0;
    for (size_t i = 0; i < predictions.size(); i++) {
        loss += (predictions[i] - targets[i]).array().square().sum();
    }
    return loss / predictions.size();
}

void trainModel(LSTMCell& lstm, const std::vector<Eigen::MatrixXf>& inputs, const std::vector<Eigen::MatrixXf>& targets, int epochs, float learningRate) {
    int count = 0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        // forward pass
        std::vector<Eigen::MatrixXf> outputs = lstm.forward(inputs);

        float loss = computeLoss(outputs, targets);
        if (epoch == count) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
            count += 1;
        }

        // compute gradients (backward pass)
        // std::vector<Eigen::MatrixXf> grad_out(outputs.size(), Eigen::MatrixXf::Zero(outputs[0].rows(), outputs[0].cols()));
        std::vector<Eigen::MatrixXf> grad_out(outputs.size());
        for (size_t t = 0; t < outputs.size(); t++) {
            grad_out[t] = 2 * (outputs[t] - targets[t]);
        }

        lstm.backpropagate(grad_out, learningRate);
    }

}