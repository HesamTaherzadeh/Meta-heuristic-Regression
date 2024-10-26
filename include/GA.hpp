#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

#include <vector>
#include <random>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>

class GeneticAlgorithm {
public:
    GeneticAlgorithm(const Eigen::MatrixXd& X, const Eigen::VectorXd& Z, int n, int m, 
                     int pop_size, int generations, double mutation_rate, int tournament_size, int patience, double coeff_lambda, double rmse_lambda);
    void run();
    void print_results();
    const std::vector<std::pair<int, int>>& getSelectedTerms() const { return selected_terms; }
    const Eigen::VectorXd& getCoefficients() const { return coeffs; }
    double getIntercept() const { return intercept; }

private:
    Eigen::MatrixXd X;
    Eigen::VectorXd Z; 
    int n;           
    int m;            
    int pop_size;
    int generations;
    int patience;
    double mutation_rate;
    int log_count;
    int tournament_size;
    int genome_length;
    std::vector<std::vector<int>> population;
    std::vector<int> best_genome;
    double best_fitness;
    std::mt19937 rng;
    double rmse_lambda, coeff_lambda;
    std::string filename;

    // Final model parameters
    std::vector<std::pair<int, int>> selected_terms;
    Eigen::VectorXd coeffs;
    double intercept;

    // Member functions
    std::vector<std::vector<int>> initial_population();
    double fitness_function(const std::vector<int>& genome);
    std::vector<int> tournament_selection(const std::vector<double>& fitnesses);
    std::pair<std::vector<int>, std::vector<int>> crossover(const std::vector<int>& parent1, const std::vector<int>& parent2);
    std::vector<int> mutate(const std::vector<int>& genome);
    void compute_final_model();
};

#endif // GENETICALGORITHM_H
