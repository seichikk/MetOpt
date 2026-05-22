#pragma once
#include <vector>
#include <string>
#include "ConstructiveReal.h"
#include "Optimization.h"
#include "Dataset.h"

ConstructiveReal forward_nn(const std::vector<double>& input, const Point& p, int input_dim, int hidden_dim);
Point numerical_gradient(ObjectiveFunction func, const Point& p, double h = 1e-5);
double calculate_f1(const Dataset& ds, int start_idx, int end_idx, const Point& p, int input_dim, int hidden_dim);
double train_and_evaluate(const std::string& filename, int input_dim, int hidden_dim, double learning_rate);
