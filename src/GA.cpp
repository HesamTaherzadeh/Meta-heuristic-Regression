#include "GA.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

GeneticAlgorithm::GeneticAlgorithm(const Eigen::VectorXd& X, const Eigen::VectorXd& Y, int n, int pop_size, int generations, double mutation_rate, int tournament_size)
    : X(X), Y(Y), n(n), pop_size(pop_size), generations(generations), mutation_rate(mutation_rate), tournament_size(tournament_size) {
    rng.seed(0);  // Set seed for reproducibility
    genome_length = n + 1;
}

void GeneticAlgorithm::run() {
    population = initial_population();
    best_fitness = std::numeric_limits<double>::infinity();

    for (int generation = 0; generation < generations; ++generation) {
        std::vector<double> fitnesses;
        for (const auto& genome : population) {
            double fitness = fitness_function(genome);
            fitnesses.push_back(fitness);
            if (fitness < best_fitness) {
                best_fitness = fitness;
                best_genome = genome;
            }
        }

        std::vector<std::vector<int>> new_population;
        while (new_population.size() < population.size()) {
            // Selection
            auto parent1 = tournament_selection(fitnesses);
            auto parent2 = tournament_selection(fitnesses);

            // Crossover
            auto offspring = crossover(parent1, parent2);
            auto child1 = mutate(offspring.first);
            new_population.push_back(child1);

            if (new_population.size() < population.size()) {
                auto child2 = mutate(offspring.second);
                new_population.push_back(child2);
            }
        }

        population = new_population;

        if (generation % 10 == 0 || generation == generations - 1) {
            std::cout << "Generation " << generation << ", Best Fitness (RMSE + penalty): " << best_fitness << std::endl;
            std::cout << "Best Genome: ";
            for (int gene : best_genome) {
                std::cout << gene;
            }
            std::cout << std::endl;
        }
    }

    // Final model training with the best genome
    compute_final_model();
}

void GeneticAlgorithm::print_results() {
    std::cout << "\nBest polynomial degrees: [";
    for (size_t i = 0; i < degrees.size(); ++i) {
        std::cout << degrees[i];
        if (i < degrees.size() - 1) std::cout << ", ";
    }
    std::cout << "]\nCoefficients: " << coeffs.transpose() << "\nIntercept: " << intercept << std::endl;
}

std::vector<std::vector<int>> GeneticAlgorithm::initial_population() {
    std::vector<std::vector<int>> population;
    std::uniform_int_distribution<int> bin_dist(0, 1);

    for (int i = 0; i < pop_size; ++i) {
        std::vector<int> genome;
        for (int j = 0; j < genome_length; ++j) {
            genome.push_back(bin_dist(rng));
        }
        population.push_back(genome);
    }
    return population;
}

double GeneticAlgorithm::fitness_function(const std::vector<int>& genome) {
    // Select polynomial degrees based on the genome
    std::vector<int> selected_degrees;
    for (int i = 0; i < genome.size(); ++i) {
        if (genome[i] == 1) {
            selected_degrees.push_back(i + 1);
        }
    }

    if (selected_degrees.empty()) {
        return 1e6;  // Penalize if no terms are selected
    }

    // Create polynomial features
    Eigen::MatrixXd X_poly(X.size(), selected_degrees.size());
    for (int i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < selected_degrees.size(); ++j) {
            X_poly(i, j) = std::pow(X(i), selected_degrees[j]);
        }
    }

    // Fit linear regression using Normal Equation
    Eigen::MatrixXd X_poly_augmented = Eigen::MatrixXd::Ones(X_poly.rows(), X_poly.cols() + 1);
    X_poly_augmented.block(0, 1, X_poly.rows(), X_poly.cols()) = X_poly;

    Eigen::VectorXd theta = (X_poly_augmented.transpose() * X_poly_augmented).ldlt().solve(X_poly_augmented.transpose() * Y);

    // Predict Y
    Eigen::VectorXd Y_pred = X_poly_augmented * theta;

    // Compute RMSE
    double rmse = std::sqrt((Y - Y_pred).squaredNorm() / Y.size());

    double penalty_factor = 0.01; 
    double penalty = penalty_factor * selected_degrees.size();

    return rmse + penalty;
}

std::vector<int> GeneticAlgorithm::tournament_selection(const std::vector<double>& fitnesses) {
    std::uniform_int_distribution<int> index_dist(0, population.size() - 1);
    std::vector<std::pair<std::vector<int>, double>> participants;

    for (int i = 0; i < tournament_size; ++i) {
        int idx = index_dist(rng);
        participants.push_back({ population[idx], fitnesses[idx] });
    }

    // Sort participants by fitness
    std::sort(participants.begin(), participants.end(), [](const std::pair<std::vector<int>, double>& a, const std::pair<std::vector<int>, double>& b) {
        return a.second < b.second;
    });

    return participants[0].first;  // Return the genome of the best participant
}

std::pair<std::vector<int>, std::vector<int>> GeneticAlgorithm::crossover(const std::vector<int>& parent1, const std::vector<int>& parent2) {
    std::uniform_int_distribution<int> point_dist(1, parent1.size() - 1);
    int point = point_dist(rng);

    std::vector<int> child1(parent1.begin(), parent1.begin() + point);
    child1.insert(child1.end(), parent2.begin() + point, parent2.end());

    std::vector<int> child2(parent2.begin(), parent2.begin() + point);
    child2.insert(child2.end(), parent1.begin() + point, parent1.end());

    return { child1, child2 };
}

std::vector<int> GeneticAlgorithm::mutate(const std::vector<int>& genome) {
    std::vector<int> mutated_genome = genome;
    std::uniform_real_distribution<double> real_dist(0.0, 1.0);

    for (size_t i = 0; i < mutated_genome.size(); ++i) {
        if (real_dist(rng) < mutation_rate) {
            mutated_genome[i] = 1 - mutated_genome[i];  // Flip bit
        }
    }
    return mutated_genome;
}

void GeneticAlgorithm::compute_final_model() {
    // Determine the degrees used in the best genome
    degrees.clear();
    for (int i = 0; i < best_genome.size(); ++i) {
        if (best_genome[i] == 1) {
            degrees.push_back(i + 1);
        }
    }

    // Create polynomial features
    Eigen::MatrixXd X_poly(X.size(), degrees.size());
    for (int i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < degrees.size(); ++j) {
            X_poly(i, j) = std::pow(X(i), degrees[j]);
        }
    }

    // Fit linear regression using Normal Equation
    Eigen::MatrixXd X_poly_augmented = Eigen::MatrixXd::Ones(X_poly.rows(), X_poly.cols() + 1);
    X_poly_augmented.block(0, 1, X_poly.rows(), X_poly.cols()) = X_poly;

    Eigen::VectorXd theta = (X_poly_augmented.transpose() * X_poly_augmented).ldlt().solve(X_poly_augmented.transpose() * Y);

    coeffs = theta.segment(1, theta.size() - 1);
    intercept = theta(0);
}
