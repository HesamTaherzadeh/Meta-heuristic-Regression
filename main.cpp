#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "GA.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Please provide the path to the YAML configuration file." << std::endl;
        return -1;
    }

    std::string yaml_file = argv[1];
    YAML::Node config = YAML::LoadFile(yaml_file);

    int data_size = config["data_size"].as<int>();
    int n = config["n"].as<int>();
    int m = config["m"].as<int>();
    int population_size = config["population_size"].as<int>();
    int generations = config["generations"].as<int>();
    double mutation_rate = config["mutation_rate"].as<double>();
    int tournament_size = config["tournament_size"].as<int>();
    int rng_seed = config["rng_seed"].as<int>();
    int patience = config["patience"].as<int>();
    double rmse_lambda = config["rmse_lambda"].as<double>();
    double coeff_lambda = config["coeff_lambda"].as<double>();



    std::mt19937 rng(rng_seed);

    std::uniform_real_distribution<double> uni_dist(-10.0, 10.0);
    std::normal_distribution<double> norm_dist(0.0, 10.0);

    Eigen::MatrixXd X_data(data_size, 2);  
    Eigen::VectorXd Z_data(data_size);

    for (int i = 0; i < data_size; ++i) {
        double xi = uni_dist(rng);
        double yi = uni_dist(rng);
        X_data(i, 0) = xi; 
        X_data(i, 1) = yi; 

        double zi = -0.2 * std::pow(xi, 3) * std::pow(yi, 5)
                    + 0.3 * std::pow(xi, 2) * yi
                    - 0.4 * xi * std::pow(yi, 2)
                    + 0.5 * xi * yi
                    + 1.0;
        zi += norm_dist(rng);
        Z_data(i) = zi;
    }

    GeneticAlgorithm ga(X_data, Z_data, n, m, population_size, generations, mutation_rate, tournament_size, patience, coeff_lambda, rmse_lambda);

    ga.run();

    ga.print_results();

    return 0;
}
