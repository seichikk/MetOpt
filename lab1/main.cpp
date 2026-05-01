#include <iostream>
#include <stdexcept>
#include "ConstructiveReal.h"
#include "Optimization.h"

ConstructiveReal rosenbrock_nd(const Point& x) {
    if (x.size() < 2) throw std::invalid_argument("нужно хотя бы 2 пространства");
    ConstructiveReal sum(0.0);
    ConstructiveReal const_1(1.0);
    ConstructiveReal const_100(100.0);

    for (size_t i = 0; i < x.size() - 1; ++i) {
        ConstructiveReal term1 = x[i + 1] - x[i].sqr();
        ConstructiveReal term2 = const_1 - x[i];
        sum = sum + (const_100 * term1.sqr()) + term2.sqr();
    }
    return sum;
}

void run_tests() {
    ConstructiveReal a(2.0, 0.1); 
    ConstructiveReal b(3.0, 0.1); 
    
    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    std::cout << "a + b = " << (a + b) << "\n";
    std::cout << "a * b = " << (a * b) << "\n";
    std::cout << "a.sqr() = " << a.sqr() << "\n\n";

    double eps = 1e-8;
    Point start_2d = { ConstructiveReal(-1.2, eps), ConstructiveReal(1.0, eps) };
    Point start_5d(5, ConstructiveReal(0.0, eps));

    GLOBAL_COMPARE_MODE = CompareMode::RELAXED;

    std::cout << "имитация отжига, min, розенброк 2D\n";
    Point res_min_2d = simulated_annealing(rosenbrock_nd, start_2d, OptGoal::MIN, 2000.0, 0.995, 10000, 0.2);
    std::cout << "точка: (" << res_min_2d[0].center() << ", " << res_min_2d[1].center() << ")\n";
    std::cout << "значение функции: " << rosenbrock_nd(res_min_2d) << "\n\n";

    std::cout << "имитация отжига, min, розенброк 5D\n";
    Point res_min_5d = simulated_annealing(rosenbrock_nd, start_5d, OptGoal::MIN, 2000.0, 0.995, 10000, 0.37);
    std::cout << "точки: (";
    for(size_t i=0; i<res_min_5d.size(); ++i) std::cout << res_min_5d[i].center() << (i == 4 ? "" : ", ");
    std::cout << ")\nзначение функции: " << rosenbrock_nd(res_min_5d) << "\n\n";

    std::cout << "случайный поиск, max, розенброк 2D\n";
    Point res_max_2d = random_search(rosenbrock_nd, 2, OptGoal::MAX, -2.0, 2.0, 50000);
    std::cout << "точка: (" << res_max_2d[0].center() << ", " << res_max_2d[1].center() << ")\n";
}

int main() {
    try {
        run_tests();
    } catch (const std::exception& e) {
        std::cerr << "ошибка: " << e.what() << "\n";
    }
    return 0;
}
