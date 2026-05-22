#pragma once
#include <vector>
#include <string>

struct Dataset {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

Dataset load_csv(const std::string& filename);
