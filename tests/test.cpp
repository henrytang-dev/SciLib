#include "../include/utils.h"
#include "../include/lstm.h"
#include "test_utils.h"
#include <assert.h>

void testXavierInit() {
    int rows = 10;
    int cols = 5;
    Eigen::MatrixXf mat(rows, cols);
    XavierInit(mat);

    float limit = std::sqrt(6.0 / (rows + cols));
    bool withinRange = (mat.array().abs() <= limit).all();

    printTestResult("Xavier Initialization", withinRange);
}

void testSigmoid() {
    LSTMCell lstm(5, 3, 10, 4);
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(2, 2);
    Eigen::MatrixXf expected = 1.0 / (1.0 + (-input.array()).exp());
    Eigen::MatrixXf result = lstm.testSigm(input);

    printTestResult("Sigmoid Function", matricesApproxEqual(result, expected));
}

void testTanh() {
    LSTMCell lstm(5, 3, 10, 4);
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(2, 2);
    Eigen::MatrixXf expected = input.array().tanh();
    Eigen::MatrixXf result = lstm.testTanh(input);

    printTestResult("Tanh Function", matricesApproxEqual(result, expected));
}

void testForwardPropagation() {
    int inputSize = 5;
    int outputSize = 3;
    int hiddenSize = 4;
    int batchSize = 2;
    int seqLength = 6;

    LSTMCell lstm(inputSize, outputSize, hiddenSize, batchSize);
    std::vector<Eigen::MatrixXf> input(seqLength, Eigen::MatrixXf::Random(inputSize, batchSize));
    
    // forward pass
    std::vector<Eigen::MatrixXf> outputs = lstm.forward(input);

    // check dimensions of outputs
    for (const auto& output : outputs) {
        assert(output.rows() == outputSize);
        assert(output.cols() == batchSize);
    }

    std::cout << "Forward propagation test passed!" << std::endl;
        
}

void testBackwardPropagation() {
    int inputSize = 5;
    int outputSize = 3;
    int hiddenSize = 4;
    int batchSize = 2;
    int seqLength = 6;
    float learningRate = 0.01;

    LSTMCell lstm(inputSize, outputSize, hiddenSize, batchSize);
    std::vector<Eigen::MatrixXf> input(seqLength, Eigen::MatrixXf::Random(inputSize, batchSize));

    std::vector<Eigen::MatrixXf> outputs = lstm.forward(input);

    std::vector<Eigen::MatrixXf> grad_out(seqLength, Eigen::MatrixXf(outputSize, batchSize));

    lstm.backpropagate(grad_out, learningRate);

    std::cout << "Backward Propagation passed!" << std::endl;
}

void testGradientDescent() {
    int inputSize = 5;
    int outputSize = 3;
    int hiddenSize = 4;
    int batchSize = 2;
    int seqLength = 6;
    float learningRate = 0.01;
    double epsilon = 1e-5;

    LSTMCell lstm(inputSize, outputSize, hiddenSize, batchSize);
    std::vector<Eigen::MatrixXf> input(seqLength, Eigen::MatrixXf::Random(inputSize, batchSize));

    std::vector<Eigen::MatrixXf> outputs = lstm.forward(input);

    std::vector<Eigen::MatrixXf> grad_out(seqLength, Eigen::MatrixXf(outputSize, batchSize));

    lstm.backpropagate(grad_out, learningRate);

    // checking gradient
    // Eigen::MatrixXf& Wf = lstm.Wf;
    // Eigen::MatrixXf numerical_grad(Wf.rows(), Wf.cols());


}