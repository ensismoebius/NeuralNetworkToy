#include "NeuralNetwork.h"

void NeuralNetwork::NeuralNetwork::initializesWeights()
{
    this->weightsInputToHidden.randu(this->hiddenNodes, this->inputNodes);
    this->weightsHiddenToOutput.randu(this->outputNodes, this->hiddenNodes);

    this->weightsInputToHidden *= 2;
    this->weightsHiddenToOutput *= 2;
    this->weightsInputToHidden -= 1;
    this->weightsHiddenToOutput -= 1;

    this->weightsInputToHidden.fill(.5);
    this->weightsHiddenToOutput.fill(.5);

    this->biasHidden.fill(.5);
    this->biasOutput.fill(.5);
}

m NeuralNetwork::NeuralNetwork::activationFunc(m value)
{
    for (auto &v : value)
    {
        v = this->activationFunction(v);
    }

    return value;
}

m NeuralNetwork::NeuralNetwork::activationFuncD(m value)
{
    for (auto &v : value)
    {
        v = this->activationFunctionDerivative(v);
    }

    return value;
}

NeuralNetwork::NeuralNetwork::NeuralNetwork(unsigned int inputNodes, unsigned int hiddenNodes, unsigned int outputNodes) :                               //
                                                                                                                           hidden(hiddenNodes, 1),       //
                                                                                                                           biasHidden(hiddenNodes, 1),   //
                                                                                                                           hiddenErrors(hiddenNodes, 1), //
                                                                                                                           output(outputNodes, 1),       //
                                                                                                                           biasOutput(outputNodes, 1),   //
                                                                                                                           outputErrors(outputNodes, 1)  //
{
    this->inputNodes = inputNodes;
    this->hiddenNodes = hiddenNodes;
    this->outputNodes = outputNodes;
    this->initializesWeights();
}

m NeuralNetwork::NeuralNetwork::classify(m input)
{
    // Generating the Hidden Outputs
    this->hidden = this->activationFunc((weightsInputToHidden * input) + this->biasHidden);

    // Generating the output's output!
    this->output = this->activationFunc((this->weightsHiddenToOutput * this->hidden) + this->biasOutput);

    // Sending back to the caller!
    return this->output;
}

void NeuralNetwork::NeuralNetwork::train(trainningSample sample, float learnningRate)
{
    //////////////////
    // Feed forward //
    //////////////////

    // Generate the hidden outputs
    // m inducedField = (weightsInputToHidden * sample.input) + biasHidden;
    // hidden = this->activationFunc(inducedField);
    this->hidden = this->activationFunc((this->weightsInputToHidden * sample.input) + this->biasHidden);

    // Generate the outputs
    // m inducedField = (weightsHiddenToOutput * hidden) + biasOutput;
    // output = this->activationFunc(inducedField);
    this->output = this->activationFunc((this->weightsHiddenToOutput * this->hidden) + this->biasOutput);

    /////////////////////
    // Backpropagation //
    /////////////////////

    m outputErrors = sample.target - this->output;

    // gradient = this->activationFuncD(output) * outputErrors * learnningRate
    // delta = gradient * hidden.tranposed
    m gradientHiddenToOutput = (this->activationFuncD(this->output) % outputErrors) * learnningRate;
    m deltaHiddenToOutput = gradientHiddenToOutput * hidden.t();
    // Update the weights
    this->weightsHiddenToOutput += deltaHiddenToOutput;
    // Update bias weights
    this->biasOutput += gradientHiddenToOutput;

    ////////////////////////////////////////////////////////////////////////

    m hiddenErrors = weightsHiddenToOutput.t() * outputErrors;

    // gradient = this->activationFuncD(hidden) * hiddenErrors * learnningRate
    // delta = gradient *sample.input.tranposed
    m gradientInputToHidden = (this->activationFuncD(this->hidden) % hiddenErrors) * learnningRate;
    m deltaInputToHidden = gradientInputToHidden * sample.input.t();
    // Update the weights
    this->weightsInputToHidden += deltaInputToHidden;
    // Update bias weights
    this->biasHidden += gradientInputToHidden;
}

void NeuralNetwork::NeuralNetwork::setActivationFunction(float (*activationFunction)(float value))
{
    this->activationFunction = activationFunction;
}

void NeuralNetwork::NeuralNetwork::setActivationFunctionDerivative(float (*activationFunctionDerivative)(float value))
{
    this->activationFunctionDerivative = activationFunctionDerivative;
}
