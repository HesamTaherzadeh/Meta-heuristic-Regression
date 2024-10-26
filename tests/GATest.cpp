// tests/GeneticAlgorithmTest.cpp

#include "GA.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

TEST(GeneticAlgorithmTest, InitializationTest) {
    // Test if the GeneticAlgorithm initializes correctly
    Eigen::MatrixXd X_data(5, 2);
    Eigen::VectorXd Z(5);
    X_data << 1, 1,
              2, 2,
              3, 3,
              4, 4,
              5, 5;
    Z << 2, 4, 6, 8, 10; // Simple linear relation: Z = 2*(x + y)

    int n = 2;
    int m = 2;
    int population_size = 10;
    int generations = 10;
    double mutation_rate = 0.01;
    int tournament_size = 3;

    GeneticAlgorithm ga(X_data, Z, n, m, population_size, generations, mutation_rate, tournament_size);
    EXPECT_NO_THROW(ga.run());
}

TEST(GeneticAlgorithmTest, CorrectnessTest) {
    // Test if the GeneticAlgorithm can recover a known polynomial in x and y
    Eigen::MatrixXd X_data(100, 2);
    Eigen::VectorXd Z(100);

    for (int i = 0; i < 100; ++i) {
        double x = i * 0.1;
        double y = i * 0.05;
        X_data(i, 0) = x;
        X_data(i, 1) = y;
        Z(i) = 3 * x * x + 2 * y + 1; // Polynomial function
    }

    int n = 2; // Maximum degree for x
    int m = 2; // Maximum degree for y
    int population_size = 50;
    int generations = 50;
    double mutation_rate = 0.001;
    int tournament_size = 3;

    GeneticAlgorithm ga(X_data, Z, n, m, population_size, generations, mutation_rate, tournament_size);
    ga.run();

     const std::vector<std::pair<int, int>>& selected_terms = ga.getSelectedTerms();
    const Eigen::VectorXd& coeffs = ga.getCoefficients();

     std::vector<std::pair<int, int>> expected_terms = { {2, 0}, {0, 1} };
    std::vector<double> true_coeffs = {3.0, 2.0};
    double true_intercept = 1.0;

    for (const auto& term : expected_terms) {
        bool includes_term = std::find(selected_terms.begin(), selected_terms.end(), term) != selected_terms.end();
        EXPECT_TRUE(includes_term) << "Term x^" << term.first << " y^" << term.second << " not found.";
    }

    for (size_t i = 0; i < expected_terms.size(); ++i) {
        const auto& term = expected_terms[i];
        auto it = std::find(selected_terms.begin(), selected_terms.end(), term);
        if (it != selected_terms.end()) {
            size_t idx = std::distance(selected_terms.begin(), it);
            EXPECT_NEAR(coeffs(idx), true_coeffs[i], 0.1) << "Coefficient for term x^" << term.first << " y^" << term.second << " is incorrect.";
        } else {
            ADD_FAILURE() << "Term x^" << term.first << " y^" << term.second << " was not found in the recovered model.";
        }
    }

    double recovered_intercept = ga.getIntercept();
    EXPECT_NEAR(recovered_intercept, true_intercept, 0.1) << "Intercept is incorrect.";
}

TEST(GeneticAlgorithmTest, HighDegreePolynomialLargeDatasetTest) {
    const int data_size = 1000; 
    Eigen::MatrixXd X_data(data_size, 2);
    Eigen::VectorXd Z(data_size);

    std::random_device rd;
    std::mt19937 rng(rd());
    rng.seed(42); 

    std::uniform_real_distribution<double> uni_dist(-10.0, 10.0);
    std::normal_distribution<double> noise_dist(0.0, 10.0);  

    // Generate data
    for (int i = 0; i < data_size; ++i) {
        double xi = uni_dist(rng);
        double yi = uni_dist(rng);
        X_data(i, 0) = xi;
        X_data(i, 1) = yi;
    }

    for (int i = 0; i < data_size; ++i) {
        double xi = X_data(i, 0);
        double yi = X_data(i, 1);
        double zi = 0.1 * std::pow(xi, 6)
                  - 0.2 * std::pow(yi, 4)
                  + 0.3 * std::pow(xi, 3)
                  - 0.4 * std::pow(yi, 2)
                  + 0.5 * xi * yi
                  + 1.0;
        zi += noise_dist(rng);
        Z(i) = zi;
    }

    int n = 6;    
    int m = 4;   
    int population_size = 100;
    int generations = 100;
    double mutation_rate = 0.1;
    int tournament_size = 3;

    GeneticAlgorithm ga(X_data, Z, n, m, population_size, generations, mutation_rate, tournament_size);
    ga.run();

    const std::vector<std::pair<int, int>>& selected_terms = ga.getSelectedTerms();
    const Eigen::VectorXd& coeffs = ga.getCoefficients();

     std::vector<std::pair<int, int>> expected_terms = {
        {6, 0}, // x^6
        {0, 4}, // y^4
        {3, 0}, // x^3
        {0, 2}, // y^2
        {1, 1}  // x^1 y^1
    };
    std::vector<double> true_coeffs = {0.1, -0.2, 0.3, -0.4, 0.5};
    double true_intercept = 1.0;

    for (const auto& term : expected_terms) {
        bool term_found = std::find(selected_terms.begin(), selected_terms.end(), term) != selected_terms.end();
        EXPECT_TRUE(term_found) << "Term x^" << term.first << " y^" << term.second << " not found in the model.";
    }

    // Check coefficients
    for (size_t i = 0; i < expected_terms.size(); ++i) {
        const auto& term = expected_terms[i];
        auto it = std::find(selected_terms.begin(), selected_terms.end(), term);
        if (it != selected_terms.end()) {
            size_t idx = std::distance(selected_terms.begin(), it);
            double recovered_coeff = coeffs(idx);
            double true_coeff = true_coeffs[i];
            EXPECT_NEAR(recovered_coeff, true_coeff, 0.25) << "Coefficient for term x^" << term.first << " y^" << term.second << " is incorrect.";
        } else {
            ADD_FAILURE() << "Term x^" << term.first << " y^" << term.second << " was not found in the recovered model.";
        }
    }

    double recovered_intercept = ga.getIntercept();
    EXPECT_NEAR(recovered_intercept, true_intercept, 0.25) << "Intercept is incorrect.";
}
