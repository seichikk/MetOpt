#pragma once
#include <vector>
#include <string>
#include "ConstructiveReal.h"
#include "Optimization.h"
#include "Dataset.h"

std::vector<ConstructiveReal> forward_nn(const std::vector<double>& input, const Point& p, int input_dim, int hidden_dim, int num_classes);
int predict_class(const std::vector<double>& input, const Point& p, int input_dim, int hidden_dim, int num_classes);
Point numerical_gradient(const ObjectiveFunction& func, const Point& p, double h = 1e-5);
double calculate_macro_f1(const Dataset& ds, int start_idx, int end_idx, const Point& p, int input_dim, int hidden_dim, int num_classes);
double train_and_evaluate(const std::string& filename, int input_dim, int hidden_dim, double learning_rate);
