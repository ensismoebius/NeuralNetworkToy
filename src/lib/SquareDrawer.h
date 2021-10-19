/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * @date 2021-10-05
 */

#ifndef src_lib_PointsDrawer_h
#define src_lib_PointsDrawer_h

#include <vector>
#include <SFML/Graphics.hpp>

#include "NeuralNetwork.h"

class SquareDrawer
{
private:
    unsigned int rows;
    unsigned int cols;
    std::vector<std::vector<sf::RectangleShape>> rectangles;

public:
    SquareDrawer(sf::RenderWindow &window, unsigned int resolution);
    void drawPoints(sf::RenderWindow &window, NeuralNetwork::NeuralNetwork &nn);
};

#endif