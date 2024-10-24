// tests/GeneticAlgorithmTest.cpp

#include "GA.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

TEST(GeneticAlgorithmTest, InitializationTest) {
    // Test if the GeneticAlgorithm initializes correctly
    Eigen::VectorXd X(5);
    Eigen::VectorXd Y(5);
    X << 1, 2, 3, 4, 5;
    Y << 2, 4, 6, 8, 10; // Simple linear relation

    int n = 2;
    int population_size = 10;
    int generations = 10;
    double mutation_rate = 0.1;
    int tournament_size = 3;

    GeneticAlgorithm ga(X, Y, n, population_size, generations, mutation_rate, tournament_size);
    EXPECT_NO_THROW(ga.run());
}

TEST(GeneticAlgorithmTest, CorrectnessTest) {
    // Test if the GeneticAlgorithm can recover a known quadratic polynomial
    Eigen::VectorXd X(100);
    Eigen::VectorXd Y(100);

    for (int i = 0; i < 100; ++i) {
        X(i) = i * 0.1;
        Y(i) = 3 * X(i) * X(i) + 2 * X(i) + 1; // Quadratic function
    }

    int n = 3;
    int population_size = 20;
    int generations = 50;
    double mutation_rate = 0.1;
    int tournament_size = 3;

    GeneticAlgorithm ga(X, Y, n, population_size, generations, mutation_rate, tournament_size);
    ga.run();

    // Access degrees and coefficients (requires adding getters)
    const std::vector<int>& degrees = ga.getDegrees();
    const Eigen::VectorXd& coeffs = ga.getCoefficients();

    // Check if degree 2 is included
    bool includes_degree_2 = std::find(degrees.begin(), degrees.end(), 2) != degrees.end();
    EXPECT_TRUE(includes_degree_2);

    // Check if the coefficients are close to the true values
    if (includes_degree_2) {
        size_t idx = std::distance(degrees.begin(), std::find(degrees.begin(), degrees.end(), 2));
        EXPECT_NEAR(coeffs(idx), 3.0, 0.1); // Allow a small margin
    }
}

TEST(GeneticAlgorithmTest, HighDegreePolynomialLargeDatasetTest) {
    // Test the algorithm with a high-degree polynomial and a large dataset
    const int data_size = 10000;  // Large dataset
    Eigen::VectorXd X(data_size);
    Eigen::VectorXd Y(data_size);

    std::random_device rd;
    std::mt19937 rng(rd());
    rng.seed(42);  // Set seed for reproducibility

    std::uniform_real_distribution<double> uni_dist(-10.0, 10.0);
    std::normal_distribution<double> noise_dist(0.0, 10.0);  // Reduced noise

    // Generate data and explicitly code the high-degree polynomial equation
    for (int i = 0; i < data_size; ++i) {
        X(i) = uni_dist(rng);
    }

    // Normalize X to [-1, 1]
    // double X_min = X.minCoeff();
    // double X_max = X.maxCoeff();
    // X = (X.array() - X_min) / (X_max - X_min) * 2.0 - 1.0;

    for (int i = 0; i < data_size; ++i) {
        double xi = X(i);
        // Explicit high-degree polynomial
        double yi = 0.1 * std::pow(xi, 6)
                  - 0.2 * std::pow(xi, 4)
                  + 0.3 * std::pow(xi, 3)
                  - 0.4 * std::pow(xi, 2)
                  + 0.5 * xi
                  + 1.0;  
        yi += noise_dist(rng); 
        Y(i) = yi;
    }

    int n = 10;  
    int population_size = 100;  
    int generations = 100;      
    double mutation_rate = 0.1; 
    int tournament_size = 3;    

    GeneticAlgorithm ga(X, Y, n, population_size, generations, mutation_rate, tournament_size);

    ga.run();

    const std::vector<int>& degrees = ga.getDegrees();
    const Eigen::VectorXd& coeffs = ga.getCoefficients();

    std::vector<int> expected_degrees = {1, 2, 3, 4, 6};  
    std::vector<double> true_coeffs = {0.5, -0.4, 0.3, -0.2, 0.1};  
    double true_intercept = 1.0;

    // Check that all expected degrees are included
    for (int degree : expected_degrees) {
        bool degree_found = std::find(degrees.begin(), degrees.end(), degree) != degrees.end();
        EXPECT_TRUE(degree_found) << "Degree " << degree << " not found in the model.";
    }

    for (size_t i = 0; i < expected_degrees.size(); ++i) {
        int degree = expected_degrees[i];
        auto it = std::find(degrees.begin(), degrees.end(), degree);
        if (it != degrees.end()) {
            size_t idx = std::distance(degrees.begin(), it);
            double recovered_coeff = coeffs(idx);
            double true_coeff = true_coeffs[i];
            EXPECT_NEAR(recovered_coeff, true_coeff, 0.1) << "Coefficient for degree " << degree << " is incorrect.";
        } else {
            ADD_FAILURE() << "Degree " << degree << " was not found in the recovered model.";
        }
    }

    double recovered_intercept = ga.getIntercept();
    EXPECT_NEAR(recovered_intercept, true_intercept, 0.1) << "Intercept is incorrect.";
}
