#ifndef MATH_H
#define MATH_H

#include <vector>
#include <utility>

double logaddexp(double a, double b);
void multiply(std::vector<double> const &x, std::vector<double> const &y, std::vector<double> &out);
std::pair<double, double> scalar_prods2(std::vector<double> const &positive1, std::vector<double> const &positive2, std::vector<double> const &x, std::vector<double> const &y);
std::pair<double, double> scalar_prods3(std::vector<double> const &positive1, std::vector<double> const &negative1, std::vector<double> const &positive2, std::vector<double> const &x, std::vector<double> const &y);
double vector_dot(std::vector<double> const &a, std::vector<double> const &b);
void axpy(std::vector<double> const &x, std::vector<double> &y, double a);
void axpy_out(std::vector<double> const &x, std::vector<double> const &y, double a, std::vector<double> &out);

#endif // MATH_H
