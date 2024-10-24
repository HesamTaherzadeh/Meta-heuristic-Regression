#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

#include <vector>
#include <random>
#include <Eigen/Dense>

class GeneticAlgorithm {
public:
    GeneticAlgorithm(const Eigen::VectorXd& X, const Eigen::VectorXd& Y, int n, int pop_size, int generations, double mutation_rate, int tournament_size);
    void run();
    void print_results();
    const std::vector<int>& getDegrees() const { return degrees; }
    const Eigen::VectorXd& getCoefficients() const { return coeffs; }
    double getIntercept() const { return intercept; }

private:
    // Member variables
    Eigen::VectorXd X;
    Eigen::VectorXd Y;
    int n;
    int pop_size;
    int generations;
    double mutation_rate;
    int tournament_size;
    int genome_length;
    std::vector<std::vector<int>> population;
    std::vector<int> best_genome;
    double best_fitness;
    std::mt19937 rng;

    // Final model parameters
    std::vector<int> degrees;
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
