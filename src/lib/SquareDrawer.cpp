#include <iostream>
#include <armadillo>
#include "SquareDrawer.h"
#include "NeuralNetwork.h"

SquareDrawer::SquareDrawer(sf::RenderWindow &window, unsigned int resolution)
{
    this->cols = window.getSize().x / resolution;
    this->rows = window.getSize().y / resolution;

    this->rectangles.resize(this->rows);
    for (auto &r : rectangles)
        r.resize(this->cols);

    for (unsigned int i = 0; i < this->cols; i++)
    {
        for (unsigned int j = 0; j < this->rows; j++)
        {
            this->rectangles[i][j] = sf::RectangleShape(sf::Vector2f(resolution, resolution));
            this->rectangles[i][j].setPosition(i * resolution, j * resolution);
            this->rectangles[i][j].setFillColor(sf::Color::White);
            window.draw(this->rectangles[i][j]);
        }
    }
}

void SquareDrawer::drawPoints(sf::RenderWindow &window, NeuralNetwork::NeuralNetwork &nn)
{

    for (unsigned int i = 0; i < this->cols; i++)
    {
        for (unsigned int j = 0; j < this->rows; j++)
        {
            float x1 = float(i) / cols;
            float x2 = float(j) / rows;
            float y = nn.classify(m({x1, x2}).t()).at(0, 0);
            this->rectangles[i][j].setFillColor(sf::Color(255 * y, 255 * y, 255 * y, 255));
            window.draw(this->rectangles[i][j]);
        }
    }
}
