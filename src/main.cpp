#include <vector>
#include <iostream>
#include <armadillo>

#include "lib/NeuralNetwork.h"

void populateExamples(std::vector<NeuralNetwork::trainningSample> &examples)
{
    examples.resize(4);

    examples[0].input = arma::Mat<float>(2, 1);
    examples[0].input.at(0, 0) = 0;
    examples[0].input.at(1, 0) = 0;
    examples[0].target = arma::Mat<float>(1, 1);
    examples[0].target.at(0, 0) = 0;

    examples[1].input = arma::Mat<float>(2, 1);
    examples[1].input.at(0, 0) = 0;
    examples[1].input.at(1, 0) = 1;
    examples[1].target = arma::Mat<float>(1, 1);
    examples[1].target.at(0, 0) = 1;

    examples[2].input = arma::Mat<float>(2, 1);
    examples[2].input.at(0, 0) = 1;
    examples[2].input.at(1, 0) = 0;
    examples[2].target = arma::Mat<float>(1, 1);
    examples[2].target.at(0, 0) = 1;

    examples[3].input = arma::Mat<float>(2, 1);
    examples[3].input.at(0, 0) = 1;
    examples[3].input.at(1, 0) = 1;
    examples[3].target = arma::Mat<float>(1, 1);
    examples[3].target.at(0, 0) = 0;
}

#define m arma::Mat<float>

void c(auto value)
{
    std::cout << value << std::endl;
}

// The activation function
arma::Mat<float> activationFunction(arma::Mat<float> value)
{
    // Apply the sigmnoid activation function for each element
    value.for_each([](float &v)
                   { v = 1.0 / (1 + std::exp(-v)); });

    return value;
}

// The derivative of activation function
arma::Mat<float> activationFunctionD(arma::Mat<float> value)
{
    return value % (1 - value);
}

// The activation function
float activationF(float value)
{
    return 1.0 / (1 + std::exp(-value));
}

// The derivative of activation function
float activationFD(float value)
{
    return value * (1 - value);
}

void initializesWeights(m &weightsInputToHidden, m &weightsHiddenToOutput, int inputNodes, int hiddenNodes, int outputNodes, m &biasHidden, m &biasOutput)
{
    weightsInputToHidden.randu(hiddenNodes, inputNodes);
    weightsHiddenToOutput.randu(outputNodes, hiddenNodes);

    weightsInputToHidden *= 2;
    weightsHiddenToOutput *= 2;
    weightsInputToHidden -= 1;
    weightsHiddenToOutput -= 1;

    weightsInputToHidden.fill(.5);
    weightsHiddenToOutput.fill(.5);

    biasHidden.fill(.5);
    biasOutput.fill(.5);
}

void train(m &input, m &target, m &weightsInputToHidden, m &hidden, m &weightsHiddenToOutput, m &output, m &biasHidden, m &biasOutput, float learnningRate)
{
    // Generate the hidden outputs
    arma::Mat<float> hiddenSums = (weightsInputToHidden * input) + biasHidden;
    hidden = activationFunction(hiddenSums);

    // Generate the outputs
    arma::Mat<float> outputSums = (weightsHiddenToOutput * hidden) + biasOutput;
    output = activationFunction(outputSums);

    // Backpropagation

    arma::Mat<float> outputErrors = target - output;
    c("Error:");
    c(outputErrors);

    // gradient = activationFunctionD(output) * outputErrors * learnningRate
    // delta = gradient * hidden.tranposed
    arma::Mat<float> gradientHiddenToOutput = (activationFunctionD(output) % outputErrors) * learnningRate;
    arma::Mat<float> deltaHiddenToOutput = gradientHiddenToOutput * hidden.t();
    // Update the weights
    weightsHiddenToOutput += deltaHiddenToOutput;
    // Update bias weights
    biasOutput += gradientHiddenToOutput;

    ////////////////////////////////////////////////////////////////////////

    arma::Mat<float> hiddenErrors = weightsHiddenToOutput.t() * outputErrors;

    // gradient = activationFunctionD(hidden) * hiddenErrors * learnningRate
    // delta = gradient * input.tranposed
    arma::Mat<float> gradientInputToHidden = (activationFunctionD(hidden) % hiddenErrors) * learnningRate;
    arma::Mat<float> deltaInputToHidden = gradientInputToHidden * input.t();
    // Update the weights
    weightsInputToHidden += deltaInputToHidden;
    // Update bias weights
    biasHidden += gradientInputToHidden;
}

int main(int argc, char const *argv[])
{
    std::vector<NeuralNetwork::trainningSample> examples;
    populateExamples(examples);

    unsigned int inputNodes = 2;
    unsigned int hiddenNodes = 4;
    unsigned int outputNodes = 1;

    m hidden(hiddenNodes, 1);
    m hiddenErrors(hiddenNodes, 1);
    m weightsInputToHidden;

    m output(outputNodes, 1);
    m outputErrors(outputNodes, 1);
    m weightsHiddenToOutput;

    m biasHidden(hiddenNodes, 1);
    m biasOutput(outputNodes, 1);

    initializesWeights(weightsInputToHidden, weightsHiddenToOutput, inputNodes, hiddenNodes, outputNodes, biasHidden, biasOutput);

    NeuralNetwork::NeuralNetwork nn(inputNodes, hiddenNodes, outputNodes);
    nn.setActivationFunction(activationF);
    nn.setActivationFunctionDerivative(activationFD);

    for (size_t i = 0; i < 40; i++)
    {
        NeuralNetwork::trainningSample e = examples[i % 4];
        train(e.input, e.target, weightsInputToHidden, hidden, weightsHiddenToOutput, output, biasHidden, biasOutput, 0.1);
        nn.train(e, 0.1);
    }
    return 0;
}
