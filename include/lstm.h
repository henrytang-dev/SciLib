#ifndef LSTM_H
#define LSTM_H

#include <Eigen/Dense>
#include <vector>

class LSTMCell {
    public:
        LSTMCell(int inputSize, int outputSize, int hiddenSize, int batchSize, float clip_value = 1.0);
        std::vector<Eigen::MatrixXf> forward(const std::vector<Eigen::MatrixXf>& inputs);
        void backpropagate(std::vector<Eigen::MatrixXf>& grad_out, float learningRate);
        void resetIntermediateValues();
        ~LSTMCell() = default;

        Eigen::MatrixXf testSigm(const Eigen::MatrixXf& x) const {
            return sigm(x);
        }

        Eigen::MatrixXf testTanh(const Eigen::MatrixXf& x) const {
            return tanh(x);
        }



    private:
        Eigen::MatrixXf Wf, Wi, Wo, Wg;
        Eigen::MatrixXf Uf, Ui, Uo, Ug;
        Eigen::MatrixXf bf, bi, bo, bg;
        Eigen::MatrixXf Wy, by;

        std::vector<Eigen::MatrixXf> f_vals, i_vals, c_tilde_vals, o_vals, c_vals, h_vals, x_vals, y_vals;

        double clip_value;
        int num_threads;
        int hiddenSize;
        int outputSize;
        int batchSize;
        int inputSize;

        Eigen::MatrixXf sigm(const Eigen::MatrixXf& x) const;
        Eigen::MatrixXf tanh(const Eigen::MatrixXf& x) const;
};

#endif // LSTM_H