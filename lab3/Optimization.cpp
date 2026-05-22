#include "Optimization.h"
#include <random>
#include <cmath>

Point random_search(ObjectiveFunction func, int dimensions, OptGoal goal,
                    double search_min, double search_max, int iterations, double initial_epsilon) 
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(search_min, search_max);

    Point best_point(dimensions);
    for (int i = 0; i < dimensions; ++i) best_point[i] = ConstructiveReal(dist(gen), initial_epsilon);
    ConstructiveReal best_val = func(best_point);

    for (int i = 0; i < iterations; ++i) {
        Point candidate(dimensions);
        for (int j = 0; j < dimensions; ++j) candidate[j] = ConstructiveReal(dist(gen), initial_epsilon);
        ConstructiveReal candidate_val = func(candidate);

        bool is_better = (goal == OptGoal::MIN) ? (candidate_val < best_val) : (candidate_val > best_val);
        if (is_better) {
            best_val = candidate_val;
            best_point = candidate;
        }
    }
    return best_point;
}

Point simulated_annealing(ObjectiveFunction func, Point start_point, OptGoal goal,
                          double initial_temp, double cooling_rate, int iterations, double step_size) 
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> rand_uniform(0.0, 1.0);
    std::normal_distribution<double> rand_normal(0.0, step_size);

    Point current_point = start_point;
    ConstructiveReal current_val = func(current_point);
    Point best_point = current_point;
    ConstructiveReal best_val = current_val;

    double temp = initial_temp;

    for (int iter = 0; iter < iterations; ++iter) {
        Point next_point = current_point;
        
        for (size_t i = 0; i < next_point.size(); ++i) {
            double step = rand_normal(gen);
            double eps = next_point[i].width() / 2.0; 
            next_point[i] = ConstructiveReal(next_point[i].center() + step, eps);
        }

        ConstructiveReal next_val = func(next_point);
        bool is_better = (goal == OptGoal::MIN) ? (next_val < current_val) : (next_val > current_val);
        
        if (is_better) {
            current_point = next_point;
            current_val = next_val;
            
            bool is_best = (goal == OptGoal::MIN) ? (current_val < best_val) : (current_val > best_val);
            if (is_best) {
                best_point = current_point;
                best_val = current_val;
            }
        } else {
            double diff = std::abs(next_val.center() - current_val.center());
            double prob = std::exp(-diff / temp);
            if (rand_uniform(gen) < prob) {
                current_point = next_point;
                current_val = next_val;
            }
        }
        temp *= cooling_rate;
    }
    return best_point;
}

Point gradient_descent(ObjectiveFunction func, GradientFunction grad, Point start_point, OptGoal goal, double learning_rate, int iterations, double tolerance) {
    Point current_point = start_point;
    ConstructiveReal lr(learning_rate);

    for (int i = 0; i < iterations; ++i) {
        Point g = grad(current_point);
        Point next_point = current_point;
        double max_step = 0.0;

        for (size_t j = 0; j < current_point.size(); ++j) {
            ConstructiveReal step = lr * g[j];
            if (goal == OptGoal::MAX) {
                next_point[j] = current_point[j] + step;
            } else {
                next_point[j] = current_point[j] - step;
            }
            max_step = std::max(max_step, std::abs(step.center()));
        }

        current_point = next_point;
        if (max_step < tolerance) break;
    }
    return current_point;
}

Point gradient_descent_momentum(ObjectiveFunction func, GradientFunction grad, Point start_point, OptGoal goal, double learning_rate, double momentum, int iterations, double tolerance) {
    Point current_point = start_point;
    Point velocity(start_point.size(), ConstructiveReal(0.0));
    ConstructiveReal lr(learning_rate);
    ConstructiveReal mom(momentum);

    for (int i = 0; i < iterations; ++i) {
        Point g = grad(current_point);
        Point next_point = current_point;
        double max_step = 0.0;

        for (size_t j = 0; j < current_point.size(); ++j) {
            velocity[j] = (mom * velocity[j]) + (lr * g[j]);
            ConstructiveReal step = velocity[j];
            
            if (goal == OptGoal::MAX) {
                next_point[j] = current_point[j] + step;
            } else {
                next_point[j] = current_point[j] - step;
            }
            max_step = std::max(max_step, std::abs(step.center()));
        }

        current_point = next_point;
        if (max_step < tolerance) break;
    }
    return current_point;
}
