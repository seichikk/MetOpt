#include "NeuralNetwork.h"
#include <cmath>
#include <algorithm>

ConstructiveReal forward_nn(const std::vector<double>& input, const Point& p, int input_dim, int hidden_dim) {
    std::vector<ConstructiveReal> hidden(hidden_dim, ConstructiveReal(0.0));
    int idx = 0;
    for (int j = 0; j < hidden_dim; ++j) {
        ConstructiveReal sum(0.0);
        for (int i = 0; i < input_dim; ++i) {
            sum = sum + (p[idx] * ConstructiveReal(input[i]));
            idx++;
        }
        sum = sum + p[idx];
        idx++;
        hidden[j] = sum.sqr();
    }
    ConstructiveReal output(0.0);
    for (int j = 0; j < hidden_dim; ++j) {
        output = output + (p[idx] * hidden[j]);
        idx++;
    }
    output = output + p[idx];
    return output;
}

Point numerical_gradient(ObjectiveFunction func, const Point& p, double h) {
    Point g(p.size(), ConstructiveReal(0.0));
    ConstructiveReal f_curr = func(p);
    for (size_t i = 0; i < p.size(); ++i) {
        Point p_pert = p;
        p_pert[i] = ConstructiveReal(p[i].center() + h, p[i].width() / 2.0);
        ConstructiveReal f_pert = func(p_pert);
        double diff = (f_pert.center() - f_curr.center()) / h;
        g[i] = ConstructiveReal(diff, 0.0, true);
    }
    return g;
}

double calculate_f1(const Dataset& ds, int start_idx, int end_idx, const Point& p, int input_dim, int hidden_dim) {
    int tp = 0, fp = 0, fn = 0;
    for (int i = start_idx; i < end_idx; ++i) {
        double pred_val = forward_nn(ds.X[i], p, input_dim, hidden_dim).center();
        int pred = (pred_val > 0.5) ? 1 : 0;
        int actual = static_cast<int>(ds.y[i]);
        if (pred == 1 && actual == 1) tp++;
        else if (pred == 1 && actual == 0) fp++;
        else if (pred == 0 && actual == 1) fn++;
    }
    if (tp + fp + fn == 0) return 0.0;
    return (2.0 * tp) / (2.0 * tp + fp + fn);
}

double train_and_evaluate(const std::string& filename, int input_dim, int hidden_dim, double learning_rate) {
    Dataset ds = load_csv(filename);
    if (ds.X.empty()) return 0.0;
    int train_size = static_cast<int>(ds.X.size() * 0.8);
    int param_count = input_dim * hidden_dim + hidden_dim + hidden_dim + 1;
    Point weights(param_count, ConstructiveReal(0.01));
    
    ObjectiveFunction loss_func = [&](const Point& p) {
        ConstructiveReal total_loss(0.0);
        for (int i = 0; i < train_size; ++i) {
            ConstructiveReal out = forward_nn(ds.X[i], p, input_dim, hidden_dim);
            ConstructiveReal target(ds.y[i]);
            total_loss = total_loss + (out - target).sqr();
        }
        return total_loss * ConstructiveReal(1.0 / train_size);
    };

    GradientFunction grad_func = [&](const Point& p) {
        return numerical_gradient(loss_func, p);
    };

    Point best_weights = gradient_descent_momentum(loss_func, grad_func, weights, OptGoal::MIN, learning_rate, 0.9, 200);
    return calculate_f1(ds, train_size, ds.X.size(), best_weights, input_dim, hidden_dim);
}
