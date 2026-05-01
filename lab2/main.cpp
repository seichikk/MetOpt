#include <iostream>
#include <stdexcept>
#include <cmath>
#include "ConstructiveReal.h"
#include "Optimization.h"

ConstructiveReal cr_sin(const ConstructiveReal& x) {
    double c = std::sin(x.center());
    double w = x.width() / 2.0;
    return ConstructiveReal(c - w, c + w, true);
}

ConstructiveReal cr_round(const ConstructiveReal& x) {
    double c = std::round(x.center());
    return ConstructiveReal(c, c, true);
}

ConstructiveReal rosenbrock_nd(const Point& x) {
    if (x.size() < 2) throw std::invalid_argument("");
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

Point rosenbrock_nd_grad(const Point& x) {
    Point g(x.size(), ConstructiveReal(0.0));
    ConstructiveReal c_2(2.0);
    ConstructiveReal c_200(200.0);
    ConstructiveReal c_400(400.0);
    ConstructiveReal c_1(1.0);

    for (size_t i = 0; i < x.size() - 1; ++i) {
        ConstructiveReal term1 = x[i+1] - x[i].sqr();
        ConstructiveReal term2 = c_1 - x[i];
        
        ConstructiveReal dx = (ConstructiveReal(0.0) - (c_400 * x[i] * term1)) - (c_2 * term2);
        ConstructiveReal dy = c_200 * term1;
        
        g[i] = g[i] + dx;
        g[i+1] = g[i+1] + dy;
    }
    return g;
}

ConstructiveReal sphere_nd(const Point& x) {
    ConstructiveReal sum(0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        sum = sum + x[i].sqr();
    }
    return sum;
}

Point sphere_nd_grad(const Point& x) {
    Point g(x.size(), ConstructiveReal(0.0));
    ConstructiveReal c_2(2.0);
    for (size_t i = 0; i < x.size(); ++i) {
        g[i] = c_2 * x[i];
    }
    return g;
}

ConstructiveReal desmos_func(const Point& pt) {
    if (pt.size() < 2) throw std::invalid_argument("");
    ConstructiveReal x = pt[0];
    ConstructiveReal y = pt[1];
    
    ConstructiveReal c_10(10.0);
    ConstructiveReal c_7(7.0);
    ConstructiveReal c_2(2.0);
    ConstructiveReal d(0.047);
    
    ConstructiveReal Ry = cr_round(cr_sin(c_10 * y)) + c_2;
    ConstructiveReal Rx = cr_round(cr_sin(c_7 * x)) + c_2;
    
    ConstructiveReal term1 = (x * Ry).sqr() + y - c_10;
    ConstructiveReal term2 = x + (y * Rx).sqr() - c_7;
    
    return d * (term1.sqr() + term2.sqr());
}

Point desmos_func_grad(const Point& pt) {
    ConstructiveReal x = pt[0];
    ConstructiveReal y = pt[1];
    
    ConstructiveReal c_10(10.0);
    ConstructiveReal c_7(7.0);
    ConstructiveReal c_2(2.0);
    ConstructiveReal d(0.047);
    ConstructiveReal c_1(1.0);
    
    ConstructiveReal Ry = cr_round(cr_sin(c_10 * y)) + c_2;
    ConstructiveReal Rx = cr_round(cr_sin(c_7 * x)) + c_2;
    
    ConstructiveReal term1 = (x * Ry).sqr() + y - c_10;
    ConstructiveReal term2 = x + (y * Rx).sqr() - c_7;
    
    ConstructiveReal dterm1_dx = c_2 * x * Ry.sqr();
    ConstructiveReal dterm2_dx = c_1;
    ConstructiveReal dx = d * (c_2 * term1 * dterm1_dx + c_2 * term2 * dterm2_dx);
    
    ConstructiveReal dterm1_dy = c_1;
    ConstructiveReal dterm2_dy = c_2 * y * Rx.sqr();
    ConstructiveReal dy = d * (c_2 * term1 * dterm1_dy + c_2 * term2 * dterm2_dy);
    
    return {dx, dy};
}

void run_tests() {
    double eps = 1e-8;
    Point start_2d = { ConstructiveReal(-1.2, eps), ConstructiveReal(1.0, eps) };
    Point start_5d(5, ConstructiveReal(2.0, eps));

    GLOBAL_COMPARE_MODE = CompareMode::RELAXED;

    std::cout << "функция розенброка 2d\n";
    Point res_sa_rosen = simulated_annealing(rosenbrock_nd, start_2d, OptGoal::MIN, 2000.0, 0.995, 10000, 0.2);
    std::cout << "имитация отжига: (" << res_sa_rosen[0].center() << ", " << res_sa_rosen[1].center() << ") -> " << rosenbrock_nd(res_sa_rosen).center() << "\n";

    Point res_gd_rosen = gradient_descent(rosenbrock_nd, rosenbrock_nd_grad, start_2d, OptGoal::MIN, 0.000001, 20000);
    std::cout << "градиентный спуск: (" << res_gd_rosen[0].center() << ", " << res_gd_rosen[1].center() << ") -> " << rosenbrock_nd(res_gd_rosen).center() << "\n";

    Point res_mom_rosen = gradient_descent_momentum(rosenbrock_nd, rosenbrock_nd_grad, start_2d, OptGoal::MIN, 0.000001, 0.9, 20000);
    std::cout << "градиентный спуск с моментом: (" << res_mom_rosen[0].center() << ", " << res_mom_rosen[1].center() << ") -> " << rosenbrock_nd(res_mom_rosen).center() << "\n\n";

    std::cout << "сферическая функция 5d\n";
    Point res_rs_sphere = random_search(sphere_nd, 5, OptGoal::MIN, -5.0, 5.0, 50000);
    std::cout << "случайный поиск (x0): " << res_rs_sphere[0].center() << " -> " << sphere_nd(res_rs_sphere).center() << "\n";

    Point res_gd_sphere = gradient_descent(sphere_nd, sphere_nd_grad, start_5d, OptGoal::MIN, 0.1, 1000);
    std::cout << "градиентный спуск (x0): " << res_gd_sphere[0].center() << " -> " << sphere_nd(res_gd_sphere).center() << "\n\n";

    std::cout << "функция desmos 2d\n";
    Point start_desmos = { ConstructiveReal(3.0, eps), ConstructiveReal(2.0, eps) };
    
    Point res_sa_desmos = simulated_annealing(desmos_func, start_desmos, OptGoal::MIN, 1000.0, 0.99, 15000, 0.5);
    std::cout << "имитация отжига: (" << res_sa_desmos[0].center() << ", " << res_sa_desmos[1].center() << ") -> " << desmos_func(res_sa_desmos).center() << "\n";

    Point res_mom_desmos = gradient_descent_momentum(desmos_func, desmos_func_grad, start_desmos, OptGoal::MIN, 0.000001, 0.8, 15000);
    std::cout << "градиентный спуск с моментом: (" << res_mom_desmos[0].center() << ", " << res_mom_desmos[1].center() << ") -> " << desmos_func(res_mom_desmos).center() << "\n";
}

int main() {
    try {
        run_tests();
    } catch (const std::exception& e) {
        std::cerr << "ошибка: " << e.what() << "\n";
    }
    return 0;
}
