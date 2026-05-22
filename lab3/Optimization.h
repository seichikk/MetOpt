#pragma once
#include "ConstructiveReal.h"
#include <vector>
#include <functional>

using Point = std::vector<ConstructiveReal>;
using ObjectiveFunction = std::function<ConstructiveReal(const Point&)>;
using GradientFunction = std::function<Point(const Point&)>;

enum class OptGoal { MIN, MAX };

Point random_search(
    ObjectiveFunction func, 
    int dimensions, 
    OptGoal goal,
    double search_min = -5.0, 
    double search_max = 5.0, 
    int iterations = 10000,
    double initial_epsilon = 1e-8
);

Point simulated_annealing(
    ObjectiveFunction func, 
    Point start_point, 
    OptGoal goal,
    double initial_temp = 1000.0, 
    double cooling_rate = 0.99, 
    int iterations = 5000,
    double step_size = 0.5
);

Point gradient_descent(
    ObjectiveFunction func, 
    GradientFunction grad, 
    Point start_point, 
    OptGoal goal, 
    double learning_rate = 0.001, 
    int iterations = 10000, 
    double tolerance = 1e-6
);

Point gradient_descent_momentum(
    ObjectiveFunction func, 
    GradientFunction grad, 
    Point start_point, 
    OptGoal goal, 
    double learning_rate = 0.001, 
    double momentum = 0.9, 
    int iterations = 10000, 
    double tolerance = 1e-6
);
