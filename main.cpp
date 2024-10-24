#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include "GA.hpp"

int main() {
    // Random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    rng.seed(0); // Set seed for reproducibility

    // Sample data generation (replace this with your actual data)
    std::uniform_real_distribution<double> uni_dist(-10.0, 10.0);
    std::normal_distribution<double> norm_dist(0.0, 10.0);

    int data_size = 10000;
    Eigen::VectorXd X(data_size);
    Eigen::VectorXd Y(data_size);

    for (int i = 0; i < data_size; ++i) {
        X(i) = uni_dist(rng);
        double xi = X(i);
        // Explicit high-degree polynomial
        double yi = - 0.2 * std::pow(xi, 4)
                  + 0.3 * std::pow(xi, 3)
                  - 0.4 * std::pow(xi, 2)
                  + 0.5 * xi
                  + 1.0;  
        yi += norm_dist(rng);  
        Y(i) = yi;
    }

    // Maximum polynomial degree
    int n = 10;

    // Genetic Algorithm parameters
    int population_size = 100;
    int generations = 50;
    double mutation_rate = 0.1;  
    int tournament_size = 3;     

    // Create an instance of GeneticAlgorithm
    GeneticAlgorithm ga(X, Y, n, population_size, generations, mutation_rate, tournament_size);

    // Run the genetic algorithm
    ga.run();

    // Print the results
    ga.print_results();

    return 0;
}
