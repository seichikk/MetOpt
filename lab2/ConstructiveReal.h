#pragma once
#include <iostream>

enum class CompareMode { STRICT, RELAXED };


extern CompareMode GLOBAL_COMPARE_MODE;

class ConstructiveReal {
private:
    double lower_bound;
    double upper_bound;

public:
    ConstructiveReal();
    ConstructiveReal(double val);
    ConstructiveReal(double val, double epsilon);
    ConstructiveReal(double min_val, double max_val, bool direct);

    double lower() const;
    double upper() const;
    double center() const;
    double width() const;

    ConstructiveReal operator+(const ConstructiveReal& other) const;
    ConstructiveReal operator-(const ConstructiveReal& other) const;
    ConstructiveReal operator*(const ConstructiveReal& other) const;
    ConstructiveReal sqr() const;

    bool operator<(const ConstructiveReal& other) const;
    bool operator>(const ConstructiveReal& other) const;

    friend std::ostream& operator<<(std::ostream& os, const ConstructiveReal& cr);
};