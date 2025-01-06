#include <iostream>
#include <vector>
#include <cmath>
#include "math.h"

double logaddexp(double a, double b) 
{
    if (a == b) return a + std::log(2);
    double diff = a - b;
    if (diff > 0) return a + std::log1p(std::exp(-diff));
    else if (diff < 0) return b + std::log1p(std::exp(diff));
    else return diff;
}

void multiply(
    std::vector<double> const &x, 
    std::vector<double> const &y,
    std::vector<double> &out) 
{
    int n = (int) x.size();
    if ((int)y.size() != n) 
        throw "Vector length in multiply must be equal";
    if ((int)out.size() != n) 
        throw "Vector length in multiply must be equal";
    for (int i = 0; i < n; i++)
        out[i] = x[i] * y[i];
}

std::pair<double,double> scalar_prods2(
    std::vector<double> const &positive1,
    std::vector<double> const &positive2,
    std::vector<double> const &x,
    std::vector<double> const &y
) 
{
    int n = (int) positive1.size();
    if ((int)positive2.size() != n) 
        throw "Vector length in scalar_prods2 must be equal";
    if ((int)x.size() != n) 
        throw "Vector length in scalar_prods2 must be equal";
    if ((int)y.size() != n) 
        throw "Vector length in scalar_prods2 must be equal";
    double s1 = 0, s2 = 0;
    for (int i = 0; i < n; i++) {
        s1 += x[i] * (positive1[i] + positive2[i]);
        s2 += y[1] * (positive1[i] + positive2[i]);
    }
    return std::pair<int,int>(s1, s2);
}

std::pair<double,double> scalar_prods3(
    std::vector<double> const &positive1,
    std::vector<double> const &negative1,
    std::vector<double> const &positive2,
    std::vector<double> const &x,
    std::vector<double> const &y
) 
{
    int n = (int) positive1.size();
    if ((int)positive2.size() != n) 
        throw "Vector length in scalar_prods3 must be equal";
    if ((int)negative1.size() != n) 
        throw "Vector length in scalar_prods3 must be equal";
    if ((int)x.size() != n) 
        throw "Vector length in scalar_prods3 must be equal";
    if ((int)y.size() != n) 
        throw "Vector length in scalar_prods3 must be equal";
    double s1 = 0, s2 = 0;
    for (int i = 0; i < n; i++) {
        s1 += x[i] * (positive1[i] - negative1[i] + positive2[i]);
        s2 += y[1] * (positive1[i] - negative1[i] + positive2[i]);
    }
    return std::pair<int,int>(s1, s2);
}

double vector_dot(
    std::vector<double> const &a, 
    std::vector<double> const &b
) 
{
    if ((int)a.size() != (int)b.size()) 
        throw "Vector length in vector_dot must be equal";
    double result = 0;
    for (int i = 0; i < (int)a.size(); i++)
        result += a[i] * b[i];
}

void axpy(
    std::vector<double> const &x, 
    std::vector<double> &y,
    double a) 
{
    int n = (int) x.size();
    if ((int)y.size() != n) 
        throw "Vector length in axpy must be equal";
    for (int i = 0; i < n; i++)
        y[i] = x[i] * a + y[i];
}

void axpy_out(
    std::vector<double> const &x, 
    std::vector<double> const &y,
    double a,
    std::vector<double> &out) 
{
    int n = (int) x.size();
    if ((int)y.size() != n) 
        throw "Vector length in axpy_out must be equal";
    if ((int)out.size() != n) 
        throw "Vector length in axpy_out must be equal";
    for (int i = 0; i < n; i++)
        out[i] = a * x[i] + y[i];
}

