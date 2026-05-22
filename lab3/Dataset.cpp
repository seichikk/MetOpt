#include "Dataset.h"
#include <fstream>
#include <sstream>

Dataset load_csv(const std::string& filename) {
    Dataset ds;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) return ds;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        if (!row.empty()) {
            ds.y.push_back(row.back());
            row.pop_back();
            ds.X.push_back(row);
        }
    }
    return ds;
}
