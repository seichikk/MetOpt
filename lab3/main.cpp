#include <iostream>
#include "ConstructiveReal.h"
#include "Optimization.h"
#include "Dataset.h"
#include "NeuralNetwork.h"

int main() {
    GLOBAL_COMPARE_MODE = CompareMode::RELAXED;
    double f1_d1 = train_and_evaluate("dataset1.csv", 2, 4, 0.1);
    double f1_d2 = train_and_evaluate("dataset2.csv", 4, 4, 0.01);
    double f1_d3 = train_and_evaluate("dataset3.csv", 4, 4, 0.01);
    
    double final_score = 0.3 * f1_d1 + 0.3 * f1_d2 + 0.4 * f1_d3;
    
    std::cout << "метрика d1: " << f1_d1 << "\n";
    std::cout << "метрика d2: " << f1_d2 << "\n";
    std::cout << "метрика d3: " << f1_d3 << "\n";
    std::cout << "итоговый результат: " << final_score << "\n";
   
    return 0;
}
