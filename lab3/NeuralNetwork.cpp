#include "NeuralNetwork.h"
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

vector<ConstructiveReal> forward_nn(const vector<double>& input, const Point& p, int input_dim, int hidden_dim, int num_classes) {
    vector<ConstructiveReal> hidden(hidden_dim, ConstructiveReal(0.0));
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
    vector<ConstructiveReal> outputs(num_classes, ConstructiveReal(0.0));
    for (int k = 0; k < num_classes; ++k) {
        ConstructiveReal out(0.0);
        for (int j = 0; j < hidden_dim; ++j) {
            out = out + (p[idx] * hidden[j]);
            idx++;
        }
        out = out + p[idx];
        idx++;
        outputs[k] = out;
    }
    return outputs;
}

int predict_class(const vector<double>& input, const Point& p, int input_dim, int hidden_dim, int num_classes) {
    vector<ConstructiveReal> scores = forward_nn(input, p, input_dim, hidden_dim, num_classes);
    int best = 0;
    double best_val = scores[0].center();
    for (int k = 1; k < num_classes; ++k) {
        double v = scores[k].center();
        if (v > best_val) {
            best_val = v;
            best = k;
        }
    }
    return best;
}

Point numerical_gradient(const ObjectiveFunction& func, const Point& p, double h) {
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

double calculate_macro_f1(const Dataset& ds, int start_idx, int end_idx, const Point& p, int input_dim, int hidden_dim, int num_classes) {
    vector<int> tp(num_classes, 0), fp(num_classes, 0), fn(num_classes, 0);
    for (int i = start_idx; i < end_idx; ++i) {
        int pred = predict_class(ds.X[i], p, input_dim, hidden_dim, num_classes);
        int actual = static_cast<int>(ds.y[i]);
        if (pred == actual) {
            tp[actual]++;
        } else {
            fp[pred]++;
            fn[actual]++;
        }
    }
    double sum_f1 = 0.0;
    int counted = 0;
    for (int k = 0; k < num_classes; ++k) {
        int denom = 2 * tp[k] + fp[k] + fn[k];
        if (denom == 0) {
            continue;
        }
        sum_f1 += (2.0 * tp[k]) / denom;
        counted++;
    }
    return counted == 0 ? 0.0 : sum_f1 / counted;
}

double train_and_evaluate(const string& filename, int input_dim, int hidden_dim, double learning_rate) {
    Dataset ds = load_csv(filename);
    if (ds.X.empty()) {
        return 0.0;
    }

    mt19937 shuffle_gen(123);
    for (int i = static_cast<int>(ds.X.size()) - 1; i > 0; --i) {
        int j = uniform_int_distribution<int>(0, i)(shuffle_gen);
        swap(ds.X[i], ds.X[j]);
        swap(ds.y[i], ds.y[j]);
    }

    int num_classes = 0;
    for (double label : ds.y) {
        num_classes = max(num_classes, static_cast<int>(label) + 1);
    }
    if (num_classes < 2) {
        num_classes = 2;
    }

    int train_size = static_cast<int>(ds.X.size() * 0.8);

    int n_features = static_cast<int>(ds.X[0].size());
    vector<double> mean(n_features, 0.0), stddev(n_features, 0.0);
    for (int i = 0; i < train_size; ++i) {
        for (int f = 0; f < n_features; ++f) {
            mean[f] += ds.X[i][f];
        }
    }
    for (int f = 0; f < n_features; ++f) {
        mean[f] /= train_size;
    }
    for (int i = 0; i < train_size; ++i) {
        for (int f = 0; f < n_features; ++f) {
            double d = ds.X[i][f] - mean[f];
            stddev[f] += d * d;
        }
    }
    for (int f = 0; f < n_features; ++f) {
        stddev[f] = sqrt(stddev[f] / train_size);
        if (stddev[f] < 1e-9) {
            stddev[f] = 1.0;
        }
    }
    for (auto& row : ds.X) {
        for (int f = 0; f < n_features; ++f) {
            row[f] = (row[f] - mean[f]) / stddev[f];
        }
    }

    int param_count = input_dim * hidden_dim + hidden_dim
                    + hidden_dim * num_classes + num_classes;

    mt19937 gen(42);
    uniform_real_distribution<double> dist(-0.8, 0.8);
    Point weights(param_count, ConstructiveReal(0.0));
    for (int i = 0; i < param_count; ++i) {
        weights[i] = ConstructiveReal(dist(gen));
    }

    ObjectiveFunction loss_func = [&](const Point& p) {
        ConstructiveReal total_loss(0.0);
        for (int i = 0; i < train_size; ++i) {
            vector<ConstructiveReal> out = forward_nn(ds.X[i], p, input_dim, hidden_dim, num_classes);
            int actual = static_cast<int>(ds.y[i]);
            for (int k = 0; k < num_classes; ++k) {
                ConstructiveReal target(k == actual ? 1.0 : 0.0);
                total_loss = total_loss + (out[k] - target).sqr();
            }
        }
        return total_loss * ConstructiveReal(1.0 / train_size);
    };

    GradientFunction grad_func = [&](const Point& p) {
        Point g = numerical_gradient(loss_func, p);
        double norm = 0.0;
        for (const auto& gi : g) {
            norm += gi.center() * gi.center();
        }
        norm = sqrt(norm);
        const double max_norm = 1.0;
        if (norm > max_norm) {
            double scale = max_norm / norm;
            for (auto& gi : g) {
                gi = ConstructiveReal(gi.center() * scale, 0.0, true);
            }
        }
        return g;
    };

    ObjectiveFunction score_func = [&](const Point& p) {
        double f1_train = calculate_macro_f1(ds, 0, train_size, p, input_dim, hidden_dim, num_classes);
        return ConstructiveReal(1.0 - f1_train);
    };

    Point best_weights = gradient_descent_momentum(score_func, grad_func, weights, OptGoal::MIN, learning_rate, 0.5, 800);
    return calculate_macro_f1(ds, train_size, ds.X.size(), best_weights, input_dim, hidden_dim, num_classes);
}
