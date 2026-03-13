#include "ConstructiveReal.h"
#include <cmath>
#include <algorithm>

CompareMode GLOBAL_COMPARE_MODE = CompareMode::STRICT;

ConstructiveReal::ConstructiveReal() : lower_bound(0.0), upper_bound(0.0) {}

ConstructiveReal::ConstructiveReal(double val) : lower_bound(val), upper_bound(val) {}

ConstructiveReal::ConstructiveReal(double val, double epsilon) 
    : lower_bound(val - epsilon), upper_bound(val + epsilon) {}

ConstructiveReal::ConstructiveReal(double min_val, double max_val, bool direct) 
    : lower_bound(min_val), upper_bound(max_val) {
    if (lower_bound > upper_bound) std::swap(lower_bound, upper_bound);
}

double ConstructiveReal::lower() const { return lower_bound; }
double ConstructiveReal::upper() const { return upper_bound; }
double ConstructiveReal::center() const { return (lower_bound + upper_bound) / 2.0; }
double ConstructiveReal::width() const { return upper_bound - lower_bound; }

ConstructiveReal ConstructiveReal::operator+(const ConstructiveReal& other) const {
    return ConstructiveReal(lower_bound + other.lower_bound, upper_bound + other.upper_bound, true);
}

ConstructiveReal ConstructiveReal::operator-(const ConstructiveReal& other) const {
    return ConstructiveReal(lower_bound - other.upper_bound, upper_bound - other.lower_bound, true);
}

ConstructiveReal ConstructiveReal::operator*(const ConstructiveReal& other) const {
    double p1 = lower_bound * other.lower_bound;
    double p2 = lower_bound * other.upper_bound;
    double p3 = upper_bound * other.lower_bound;
    double p4 = upper_bound * other.upper_bound;
    return ConstructiveReal(std::min({p1, p2, p3, p4}), std::max({p1, p2, p3, p4}), true);
}

ConstructiveReal ConstructiveReal::sqr() const {
    if (lower_bound <= 0.0 && upper_bound >= 0.0) {
        double max_sqr = std::max(lower_bound * lower_bound, upper_bound * upper_bound);
        return ConstructiveReal(0.0, max_sqr, true);
    }
    double p1 = lower_bound * lower_bound;
    double p2 = upper_bound * upper_bound;
    return ConstructiveReal(std::min(p1, p2), std::max(p1, p2), true);
}

bool ConstructiveReal::operator<(const ConstructiveReal& other) const {
    if (GLOBAL_COMPARE_MODE == CompareMode::STRICT) {
        return upper_bound < other.lower_bound;
    } else {
        return center() < other.center();
    }
}

bool ConstructiveReal::operator>(const ConstructiveReal& other) const {
    return other < *this;
}

std::ostream& operator<<(std::ostream& os, const ConstructiveReal& cr) {
    os << "[" << cr.lower_bound << ", " << cr.upper_bound << "] (c: " << cr.center() << ")";
    return os;
}