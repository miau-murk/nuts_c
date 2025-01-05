#include <gtest/gtest.h>
#include "math.h"
#include <limits>
#include <random>
#include <algorithm>

// Тест для функции logaddexp
TEST(LogAddExpTest, RandomValues) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    for (int i = 0; i < 100; ++i) {
        double a = dis(gen);
        double b = dis(gen);
        EXPECT_DOUBLE_EQ(logaddexp(a, b), std::log(std::exp(a) + std::exp(b)));
    }
}

TEST(LogAddExpTest, EdgeCases) {
    EXPECT_DOUBLE_EQ(logaddexp(0.0, 0.0), std::log(2));
    EXPECT_DOUBLE_EQ(logaddexp(std::numeric_limits<double>::infinity(), 0.0), std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(logaddexp(0.0, std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(logaddexp(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity());
}

// Тест для функции multiply
TEST(MultiplyTest, RandomValues) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    std::vector<double> x(10), y(10), out(10);
    for (int i = 0; i < 10; ++i) {
        x[i] = dis(gen);
        y[i] = dis(gen);
    }

    multiply(x, y, out);

    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(out[i], x[i] * y[i]);
    }
}

TEST(MultiplyTest, EdgeCases) {
    std::vector<double> x = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> y = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> out(4);

    multiply(x, y, out);

    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_DOUBLE_EQ(out[1], 1.0);
    EXPECT_DOUBLE_EQ(out[2], -1.0);
    EXPECT_DOUBLE_EQ(out[3], std::numeric_limits<double>::infinity());
}

// Тест для функции scalar_prods2
TEST(ScalarProds2Test, RandomValues) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    std::vector<double> positive1(10), positive2(10), x(10), y(10);
    for (int i = 0; i < 10; ++i) {
        positive1[i] = dis(gen);
        positive2[i] = dis(gen);
        x[i] = dis(gen);
        y[i] = dis(gen);
    }

    auto result = scalar_prods2(positive1, positive2, x, y);
    double expected_s1 = 0, expected_s2 = 0;
    for (int i = 0; i < 10; ++i) {
        expected_s1 += x[i] * (positive1[i] + positive2[i]);
        expected_s2 += y[i] * (positive1[i] + positive2[i]);
    }

    EXPECT_DOUBLE_EQ(result.first, expected_s1);
    EXPECT_DOUBLE_EQ(result.second, expected_s2);
}

TEST(ScalarProds2Test, EdgeCases) {
    std::vector<double> positive1 = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> positive2 = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> x = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> y = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};

    auto result = scalar_prods2(positive1, positive2, x, y);

    EXPECT_DOUBLE_EQ(result.first, std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(result.second, std::numeric_limits<double>::infinity());
}

// Тест для функции scalar_prods3
TEST(ScalarProds3Test, RandomValues) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    std::vector<double> positive1(10), negative1(10), positive2(10), x(10), y(10);
    for (int i = 0; i < 10; ++i) {
        positive1[i] = dis(gen);
        negative1[i] = dis(gen);
        positive2[i] = dis(gen);
        x[i] = dis(gen);
        y[i] = dis(gen);
    }

    auto result = scalar_prods3(positive1, negative1, positive2, x, y);
    double expected_s1 = 0, expected_s2 = 0;
    for (int i = 0; i < 10; ++i) {
        expected_s1 += x[i] * (positive1[i] - negative1[i] + positive2[i]);
        expected_s2 += y[i] * (positive1[i] - negative1[i] + positive2[i]);
    }

    EXPECT_DOUBLE_EQ(result.first, expected_s1);
    EXPECT_DOUBLE_EQ(result.second, expected_s2);
}

TEST(ScalarProds3Test, EdgeCases) {
    std::vector<double> positive1 = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> negative1 = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> positive2 = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> x = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> y = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};

    auto result = scalar_prods3(positive1, negative1, positive2, x, y);

    EXPECT_DOUBLE_EQ(result.first, std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(result.second, std::numeric_limits<double>::infinity());
}

// Тест для функции vector_dot
TEST(VectorDotTest, RandomValues) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    std::vector<double> a(10), b(10);
    for (int i = 0; i < 10; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    double result = vector_dot(a, b);
    double expected_result = 0;
    for (int i = 0; i < 10; ++i) {
        expected_result += a[i] * b[i];
    }

    EXPECT_DOUBLE_EQ(result, expected_result);
}

TEST(VectorDotTest, EdgeCases) {
    std::vector<double> a = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> b = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};

    double result = vector_dot(a, b);

    EXPECT_DOUBLE_EQ(result, std::numeric_limits<double>::infinity());
}

// Тест для функции axpy
TEST(AxpyTest, RandomValues) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    std::vector<double> x(10), y(10);
    double a = dis(gen);
    for (int i = 0; i < 10; ++i) {
        x[i] = dis(gen);
        y[i] = dis(gen);
    }

    axpy(x, y, a);

    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i] * a + y[i]);
    }
}

TEST(AxpyTest, EdgeCases) {
    std::vector<double> x = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> y = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    double a = 2.0;

    axpy(x, y, a);

    EXPECT_DOUBLE_EQ(y[0], 0.0);
    EXPECT_DOUBLE_EQ(y[1], 3.0);
    EXPECT_DOUBLE_EQ(y[2], -3.0);
    EXPECT_DOUBLE_EQ(y[3], std::numeric_limits<double>::infinity());
}

// Тест для функции axpy_out
TEST(AxpyOutTest, RandomValues) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    std::vector<double> x(10), y(10), out(10);
    double a = dis(gen);
    for (int i = 0; i < 10; ++i) {
        x[i] = dis(gen);
        y[i] = dis(gen);
    }

    axpy_out(x, y, a, out);

    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(out[i], a * x[i] + y[i]);
    }
}

TEST(AxpyOutTest, EdgeCases) {
    std::vector<double> x = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> y = {0.0, 1.0, -1.0, std::numeric_limits<double>::infinity()};
    std::vector<double> out(4);
    double a = 2.0;

    axpy_out(x, y, a, out);

    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_DOUBLE_EQ(out[1], 3.0);
    EXPECT_DOUBLE_EQ(out[2], -3.0);
    EXPECT_DOUBLE_EQ(out[3], std::numeric_limits<double>::infinity());
}
