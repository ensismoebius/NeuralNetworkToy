#include <vector>
#include <iostream>
#include <armadillo>
#include <SFML/Graphics.hpp>

#include "lib/SquareDrawer.h"
#include "lib/NeuralNetwork.h"

////////////////////////////////////////////
///////////////// SETTINGS /////////////////
////////////////////////////////////////////
const int SQUARE_SIZE = 50;
const float WINDOW_WIDTH = 800;
const float WINDOW_HEIGHT = 800;
const float TRAINING_RATE = 0.001;
const char FONT_PATH[] = "../Arial.ttf";
////////////////////////////////////////////
////////////// SETTINGS - END //////////////
////////////////////////////////////////////

void populateExamples(std::vector<NeuralNetwork::trainningSample> &examples)
{
    examples.resize(17);

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

    examples[4].input = arma::Mat<float>(2, 1);
    examples[4].input.at(0, 0) = 4;
    examples[4].input.at(1, 0) = 4;
    examples[4].target = arma::Mat<float>(1, 1);
    examples[4].target.at(0, 0) = 0;

    examples[5].input = arma::Mat<float>(2, 1);
    examples[5].input.at(0, 0) = 7;
    examples[5].input.at(1, 0) = 0;
    examples[5].target = arma::Mat<float>(1, 1);
    examples[5].target.at(0, 0) = 1;

    examples[6].input = arma::Mat<float>(2, 1);
    examples[6].input.at(0, 0) = 0;
    examples[6].input.at(1, 0) = 7;
    examples[6].target = arma::Mat<float>(1, 1);
    examples[6].target.at(0, 0) = 1;

    examples[7].input = arma::Mat<float>(2, 1);
    examples[7].input.at(0, 0) = 3;
    examples[7].input.at(1, 0) = 2;
    examples[7].target = arma::Mat<float>(1, 1);
    examples[7].target.at(0, 0) = 1;

    examples[8].input = arma::Mat<float>(2, 1);
    examples[8].input.at(0, 0) = 2;
    examples[8].input.at(1, 0) = 3;
    examples[8].target = arma::Mat<float>(1, 1);
    examples[8].target.at(0, 0) = 1;

    examples[9].input = arma::Mat<float>(2, 1);
    examples[9].input.at(0, 0) = 6;
    examples[9].input.at(1, 0) = 7;
    examples[9].target = arma::Mat<float>(1, 1);
    examples[9].target.at(0, 0) = 1;

    examples[10].input = arma::Mat<float>(2, 1);
    examples[10].input.at(0, 0) = 2;
    examples[10].input.at(1, 0) = 1;
    examples[10].target = arma::Mat<float>(1, 1);
    examples[10].target.at(0, 0) = 1;

    examples[11].input = arma::Mat<float>(2, 1);
    examples[11].input.at(0, 0) = 1;
    examples[11].input.at(1, 0) = 2;
    examples[11].target = arma::Mat<float>(1, 1);
    examples[11].target.at(0, 0) = 1;

    examples[12].input = arma::Mat<float>(2, 1);
    examples[12].input.at(0, 0) = 7;
    examples[12].input.at(1, 0) = 6;
    examples[12].target = arma::Mat<float>(1, 1);
    examples[12].target.at(0, 0) = 1;

    examples[13].input = arma::Mat<float>(2, 1);
    examples[13].input.at(0, 0) = 0;
    examples[13].input.at(1, 0) = 2;
    examples[13].target = arma::Mat<float>(1, 1);
    examples[13].target.at(0, 0) = 1;

    examples[14].input = arma::Mat<float>(2, 1);
    examples[14].input.at(0, 0) = 2;
    examples[14].input.at(1, 0) = 0;
    examples[14].target = arma::Mat<float>(1, 1);
    examples[14].target.at(0, 0) = 1;

    examples[15].input = arma::Mat<float>(2, 1);
    examples[15].input.at(0, 0) = 0;
    examples[15].input.at(1, 0) = 4;
    examples[15].target = arma::Mat<float>(1, 1);
    examples[15].target.at(0, 0) = 1;

    examples[16].input = arma::Mat<float>(2, 1);
    examples[16].input.at(0, 0) = 4;
    examples[16].input.at(1, 0) = 0;
    examples[16].target = arma::Mat<float>(1, 1);
    examples[16].target.at(0, 0) = 1;
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

int main(int argc, char const *argv[])
{
    /////////////////////////////
    // Window creation - BEGIN //
    /////////////////////////////

    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;

    int32_t windowStyle = sf::Style::Default;

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Neural Network - float", windowStyle, settings);
    window.setFramerateLimit(0);

    // Puts the orign at the center
    sf::Vector2u size = window.getSize();
    sf::View view(sf::Vector2f(size.x / 2, size.y / 2), sf::Vector2f(WINDOW_WIDTH, WINDOW_HEIGHT));
    window.setView(view);

    ///////////////////////////
    // Window creation - END //
    ///////////////////////////

    // Ensuring that the random numbers are going to honor its names
    std::srand(std::time(nullptr));

    std::vector<NeuralNetwork::trainningSample> examples;
    populateExamples(examples);

    NeuralNetwork::NeuralNetwork nn(2, 40, 1);
    nn.setActivationFunction(activationF);
    nn.setActivationFunctionDerivative(activationFD);

    sf::Font font;
    font.loadFromFile(FONT_PATH);
    SquareDrawer sd(window, SQUARE_SIZE, font);

    auto rng = std::default_random_engine{};
    // Aplication main loop
    while (window.isOpen())
    {

        std::shuffle(examples.begin(), examples.end(), rng);
        for (size_t i = 0; i < 40; i++)
        {
            // Neural network tranning
            for (auto e : examples)
            {
                nn.train(e, TRAINING_RATE);
            }
        }

        // Drawing the data
        window.clear(sf::Color::Magenta);
        sd.drawPoints(window, nn);
        window.display();

        // Window and keyboard events
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Q)
                window.close();
        }
    }

    return 0;
}
