#include <vector>
#include <iostream>
#include <armadillo>

struct example
{
    arma::Mat<float> input;
    arma::Mat<float> target;
};

void populateExamples(std::vector<example> &examples)
{
    examples.resize(2);

    examples[0].input = arma::Mat<float>(2, 1);
    examples[0].input.at(0, 0) = 1;
    examples[0].input.at(1, 0) = 1;

    examples[0].target = arma::Mat<float>(1, 1);
    examples[0].target.at(0, 0) = 1;

    examples[1].input = arma::Mat<float>(2, 1);
    examples[1].input.at(0, 0) = 1;
    examples[1].input.at(1, 0) = 0;

    examples[1].target = arma::Mat<float>(1, 1);
    examples[1].target.at(0, 0) = 0;
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
    value = activationFunction(value) % (1 - activationFunction(value));

    return value;
}

void initializesWeights(m &weightsInputToHidden, m &weightsHiddenToOutput, int inputNodes, int hiddenNodes, int outputNodes)
{
    weightsInputToHidden.randu(hiddenNodes, inputNodes);
    weightsHiddenToOutput.randu(outputNodes, hiddenNodes);

    weightsInputToHidden *= 2;
    weightsHiddenToOutput *= 2;
    weightsInputToHidden -= 1;
    weightsHiddenToOutput -= 1;

    weightsInputToHidden.fill(.5);
    weightsHiddenToOutput.fill(.5);
}

void feedFoward(m &input, m &target, m &weightsInputToHidden, m &hidden, m &weightsHiddenToOutput, m &output, float learnningRate)
{
    // Generate the hidden outputs
    arma::Mat<float> hiddenSums = weightsInputToHidden * input;
    hidden = activationFunction(hiddenSums);

    // Generate the outputs
    arma::Mat<float> outputSums = weightsHiddenToOutput * hidden;
    output = activationFunction(outputSums);

    // Backprop

    arma::Mat<float> outputErrors = target - output;
    outputErrors = (outputErrors % outputErrors) * 0.5;

    c("Errors:");
    c(outputErrors);

    arma::Mat<float> gradient = outputErrors * activationFunctionD(output);
    c("Gradient:");
    c(gradient);

    arma::Mat<float> weightCorretions = learnningRate * gradient * hidden;
    c("Weight corrections:");
    c(weightCorretions);

    c("Old weights:");
    c(weightsHiddenToOutput);

    weightsHiddenToOutput = weightsHiddenToOutput + weightCorretions;

    c("Updated weights:");
    c(weightCorretions);
}

int main(int argc, char const *argv[])
{
    std::vector<example> examples;
    populateExamples(examples);

    unsigned int inputNodes = 2;
    unsigned int hiddenNodes = 2;
    unsigned int outputNodes = 1;

    m hidden(hiddenNodes, 1);
    m hiddenErrors(hiddenNodes, 1);
    m weightsInputToHidden;

    m output(outputNodes, 1);
    m outputErrors(outputNodes, 1);
    m weightsHiddenToOutput;

    initializesWeights(weightsInputToHidden, weightsHiddenToOutput, inputNodes, hiddenNodes, outputNodes);

    for (auto e : examples)
    {
        feedFoward(e.input, e.target, weightsInputToHidden, hidden, weightsHiddenToOutput, output, 0.1);
    }

    return 0;
}
