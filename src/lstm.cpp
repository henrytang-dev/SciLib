#include <iostream>
#include "../include/lstm.h"
#include "../include/utils.h"
#include <omp.h>

LSTMCell::LSTMCell(int inputSize, int outputSize, int hiddenSize, int batchSize, float clip_value) : inputSize(inputSize), batchSize(batchSize), outputSize(outputSize), hiddenSize(hiddenSize), clip_value(clip_value), num_threads(0) {
    XavierInit(Wf = Eigen::MatrixXf(hiddenSize, inputSize));
    XavierInit(Wi = Eigen::MatrixXf(hiddenSize, inputSize));
    XavierInit(Wo = Eigen::MatrixXf(hiddenSize, inputSize));
    XavierInit(Wg = Eigen::MatrixXf(hiddenSize, inputSize));

    XavierInit(Uf = Eigen::MatrixXf(hiddenSize, hiddenSize));
    XavierInit(Ui = Eigen::MatrixXf(hiddenSize, hiddenSize));
    XavierInit(Uo = Eigen::MatrixXf(hiddenSize, hiddenSize));
    XavierInit(Ug = Eigen::MatrixXf(hiddenSize, hiddenSize));

    XavierInit(Wy = Eigen::MatrixXf(outputSize, hiddenSize));
    by = Eigen::VectorXf::Zero(outputSize);

    bf = Eigen::VectorXf::Zero(hiddenSize);
    bi = Eigen::VectorXf::Zero(hiddenSize);
    bo = Eigen::VectorXf::Zero(hiddenSize);
    bg = Eigen::VectorXf::Zero(hiddenSize);
}

Eigen::MatrixXf LSTMCell::sigm(const Eigen::MatrixXf& x) const {
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
}

Eigen::MatrixXf LSTMCell::tanh(const Eigen::MatrixXf& x) const {
    return x.array().tanh().matrix();
}

// LSTMCell::~LSTMCell() {}

void LSTMCell::resetIntermediateValues() {
    f_vals.clear();
    i_vals.clear();
    c_tilde_vals.clear();
    o_vals.clear();
    c_vals.clear();
    h_vals.clear();
    x_vals.clear();
    y_vals.clear();
}

// think of all the hidden states running in parallel
std::vector<Eigen::MatrixXf> LSTMCell::forward(const std::vector<Eigen::MatrixXf>& inputs) {
    // inputs: ()
    int batchSize = inputs[0].cols(); // batch number/ # of sequences
    int seqLength = inputs.size(); // timesteps

    resetIntermediateValues();

    f_vals.resize(seqLength, Eigen::MatrixXf::Zero(hiddenSize, batchSize));
    i_vals.resize(seqLength, Eigen::MatrixXf::Zero(hiddenSize, batchSize));

    c_tilde_vals.resize(seqLength, Eigen::MatrixXf::Zero(hiddenSize, batchSize));

    o_vals.resize(seqLength, Eigen::MatrixXf::Zero(hiddenSize, batchSize));
    c_vals.resize(seqLength, Eigen::MatrixXf::Zero(hiddenSize, batchSize)); // Uf.rows() = hiddenSize
    h_vals.resize(seqLength, Eigen::MatrixXf::Zero(hiddenSize, batchSize)); // Uf.rows() = hiddenSize
    x_vals.resize(seqLength, Eigen::MatrixXf::Zero(inputSize, batchSize));
    y_vals.resize(seqLength, Eigen::MatrixXf::Zero(outputSize, batchSize));

    std::vector<Eigen::MatrixXf> outputs(seqLength, Eigen::MatrixXf::Zero(inputs[0].rows(), batchSize)); // do i really need this?

    for (int t = 0; t < seqLength; t++) {
        const Eigen::MatrixXf& x_t = inputs[t];

        #pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            Eigen::MatrixXf x_b = x_t.col(b);

            Eigen::MatrixXf h_b = h_vals[t].col(b); // hidden states of this sequence
            Eigen::MatrixXf c_b = c_vals[t].col(b); // cell state of this independent sequence

            Eigen::MatrixXf f, i, c_tilde, o;

            f = sigm(Wf * x_b + Uf * h_b + bf);
            i = sigm(Wi * x_b + Ui * h_b + bi);
            c_tilde = tanh(Wg * x_b + Ug * h_b + bg);
            o = sigm(Wo * x_b + Uo * h_b + bo);


            c_b = f.cwiseProduct(c_b) + i.cwiseProduct(c_tilde);
            h_b = o.cwiseProduct(tanh(c_b));

            f_vals[t].col(b) = f;
            i_vals[t].col(b) = i;
            c_tilde_vals[t].col(b) = c_tilde;
            o_vals[t].col(b) = o;
            c_vals[t].col(b) = c_b;
            h_vals[t].col(b) = h_b;
            x_vals[t].col(b) = x_b;
        }

        Eigen::MatrixXf y_t = Wy * h_vals[t];
        // y_t.colwise() += by;


        #pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            y_vals[t].col(b) = y_t.col(b) += by;
        }

        outputs[t] = y_t;
    }
    return outputs;
}

void LSTMCell::backpropagate(std::vector<Eigen::MatrixXf>& grad_out, float learningRate) {
    int batchSize = grad_out[0].cols();
    int seqLength =  grad_out.size();

    Eigen::MatrixXf dWy = Eigen::MatrixXf::Zero(Wy.rows(), Wy.cols());
    Eigen::MatrixXf dby = Eigen::MatrixXf::Zero(by.rows(), by.cols());

    Eigen::MatrixXf dWf = Eigen::MatrixXf::Zero(Wf.rows(), Wf.cols());
    Eigen::MatrixXf dWi = Eigen::MatrixXf::Zero(Wi.rows(), Wi.cols());
    Eigen::MatrixXf dWo = Eigen::MatrixXf::Zero(Wo.rows(), Wo.cols());
    Eigen::MatrixXf dWg = Eigen::MatrixXf::Zero(Wg.rows(), Wg.cols());

    Eigen::MatrixXf dUf = Eigen::MatrixXf::Zero(Uf.rows(), Uf.cols());
    Eigen::MatrixXf dUi = Eigen::MatrixXf::Zero(Ui.rows(), Ui.cols());
    Eigen::MatrixXf dUo = Eigen::MatrixXf::Zero(Uo.rows(), Uo.cols());
    Eigen::MatrixXf dUg = Eigen::MatrixXf::Zero(Ug.rows(), Ug.cols());

    Eigen::MatrixXf dbf = Eigen::MatrixXf::Zero(bf.rows(), bf.cols());
    Eigen::MatrixXf dbi = Eigen::MatrixXf::Zero(bi.rows(), bi.cols());
    Eigen::MatrixXf dbo = Eigen::MatrixXf::Zero(bo.rows(), bo.cols());
    Eigen::MatrixXf dbg = Eigen::MatrixXf::Zero(bg.rows(), bg.cols());

    Eigen::MatrixXf dc_next = Eigen::MatrixXf::Zero(hiddenSize, batchSize);
    Eigen::MatrixXf dh_next = Eigen::MatrixXf::Zero(hiddenSize, batchSize);

    for (int t = seqLength - 1; t >= 0; t--) {
        const Eigen::MatrixXf& grad_output_t = grad_out[t];

        // #pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            Eigen::MatrixXf dh_total = Wy.transpose() * grad_output_t.col(b) + dh_next.col(b);
            Eigen::MatrixXf dc = dh_total.cwiseProduct(o_vals[t].col(b)).cwiseProduct((1 - tanh(c_vals[t].col(b)).array().square()).matrix()) + dc_next.col(b);

            // Derivatives are in respect to activations
            Eigen::MatrixXf d_o = dh_total.cwiseProduct(tanh(c_vals[t].col(b))).cwiseProduct(o_vals[t].col(b).cwiseProduct((1 - o_vals[t].col(b).array()).matrix()));
            Eigen::MatrixXf di = dc.cwiseProduct(c_tilde_vals[t].col(b)).cwiseProduct(i_vals[t].col(b).cwiseProduct((1 - i_vals[t].col(b).array()).matrix()));
            Eigen::MatrixXf df = Eigen::MatrixXf::Zero(hiddenSize, 1);
            if (t != 0) {
                df = dc.cwiseProduct(c_vals[t - 1].col(b)).cwiseProduct(f_vals[t].col(b).cwiseProduct((1 - f_vals[t].col(b).array()).matrix())); // i think edge case for final reversal time sequence
            }
            Eigen::MatrixXf dc_tilde = dc.cwiseProduct(i_vals[t].col(b)).cwiseProduct((1 - c_tilde_vals[t].col(b).array().square()).matrix());


            dWo += d_o * x_vals[t].col(b).transpose();
            dUo += d_o * h_vals[t].col(b).transpose();
            dbo += d_o;

            dWg += dc_tilde * x_vals[t].col(b).transpose();
            dUg += dc_tilde * h_vals[t].col(b).transpose();
            dbg += dc_tilde;

            dWf += df * x_vals[t].col(b).transpose();
            dUf += df * h_vals[t].col(b).transpose();
            dbf += df;

            dWi += di * x_vals[t].col(b).transpose();
            dUi += di * h_vals[t].col(b).transpose();
            dbi += di;

            dh_next.col(b) = Uf.transpose() * df + Ui.transpose() * di + Uo.transpose() * d_o + Ug.transpose() * dc_tilde;
            dc_next.col(b) = dc.cwiseProduct(f_vals[t].col(b));

            dWy += grad_output_t.col(b) * h_vals[t].col(b).transpose();
            dby += grad_output_t.col(b);
        }
    }
    
    Wf -= learningRate * dWf;
    Wi -= learningRate * dWi;
    Wo -= learningRate * dWo;
    Wg -= learningRate * dWg;

    Uf -= learningRate * dUf;
    Ui -= learningRate * dUi;
    Uo -= learningRate * dUo;
    Ug -= learningRate * dUg;

    bf -= learningRate * dbf;
    bi -= learningRate * dbi;
    bo -= learningRate * dbo;
    bg -= learningRate * dbg;

    Wy -= learningRate * dWy;
    by -= learningRate * dby;
}