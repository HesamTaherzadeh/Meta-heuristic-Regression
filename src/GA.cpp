#include "GA.hpp"

std::string getCurrentTimeForFilename() {
    auto now = std::chrono::system_clock::now();
    std::time_t time_now = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&time_now);

    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

GeneticAlgorithm::GeneticAlgorithm(const Eigen::MatrixXd& X, const Eigen::VectorXd& Z, int n, int m, 
                                   int pop_size, int generations, double mutation_rate, int tournament_size, int patience, double coeff_lambda, double rmse_lambda)
    : X(X), Z(Z), n(n), m(m), pop_size(pop_size), generations(generations), mutation_rate(mutation_rate), 
      tournament_size(tournament_size) , patience(patience), coeff_lambda(coeff_lambda), rmse_lambda(rmse_lambda){
    rng.seed(0);   
    genome_length = (n + 1) * (m + 1);  
    log_count = 0;
    filename = "logs/fitness_log_" + getCurrentTimeForFilename() + ".csv";

}

void GeneticAlgorithm::run() {
    population = initial_population();
    best_fitness = std::numeric_limits<double>::infinity();
    
    double improvement_threshold = 1e-6;  
    int patience_counter = 0;  

    for (int generation = 0; generation < generations; ++generation) {
        std::vector<double> fitnesses;
        bool fitness_improved = false;

        for (const auto& genome : population) {
            double fitness = fitness_function(genome);
            fitnesses.push_back(fitness);
            if (fitness < best_fitness) {
                if (best_fitness - fitness > improvement_threshold) {
                    fitness_improved = true;
                }
                best_fitness = fitness;
                best_genome = genome;
            }
        }

        std::vector<std::vector<int>> new_population;
        while (new_population.size() < population.size()) {
            auto parent1 = tournament_selection(fitnesses);
            auto parent2 = tournament_selection(fitnesses);

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

        if (fitness_improved) {
            patience_counter = 0;  
        } else {
            patience_counter++;
        }

        if (patience_counter >= patience) {
            std::cout << "Early stopping triggered after " << generation + 1 << " generations." << std::endl;
            break;  
        }
    }

    compute_final_model();
}


void GeneticAlgorithm::print_results() {
    std::cout << "\nSelected terms (x^i y^j):\n";
    for (const auto& term : selected_terms) {
        std::cout << "x^" << term.first << " y^" << term.second << "\n";
    }
    std::cout << "Coefficients: " << coeffs.transpose() << "\nIntercept: " << intercept << std::endl;
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
    std::vector<std::pair<int, int>> selected_terms_local;
    for (int idx = 0; idx < genome.size(); ++idx) {
        if (genome[idx] == 1) {
            int i = idx / (m + 1);
            int j = idx % (m + 1);
            selected_terms_local.emplace_back(i, j);
        }
    }

    if (selected_terms_local.empty()) {
        return 1e6;  
    }

    // Create polynomial features
    Eigen::MatrixXd X_poly(X.rows(), selected_terms_local.size());
    for (int sample = 0; sample < X.rows(); ++sample) {
        for (size_t idx = 0; idx < selected_terms_local.size(); ++idx) {
            int i = selected_terms_local[idx].first;
            int j = selected_terms_local[idx].second;
            X_poly(sample, idx) = std::pow(X(sample, 0), i) * std::pow(X(sample, 1), j);
        }
    }

    Eigen::MatrixXd X_poly_augmented = Eigen::MatrixXd::Ones(X_poly.rows(), X_poly.cols() + 1);
    X_poly_augmented.block(0, 1, X_poly.rows(), X_poly.cols()) = X_poly;

    Eigen::VectorXd theta = (X_poly_augmented.transpose() * X_poly_augmented).ldlt().solve(X_poly_augmented.transpose() * Z);

    Eigen::VectorXd Z_pred = X_poly_augmented * theta;

    double rmse = std::sqrt((Z - Z_pred).squaredNorm() / Z.size());

    double penalty = rmse_lambda * selected_terms_local.size();

    double epsilon = 1e-3;                   
    int small_coeff_count = 0;

    for (int k = 1; k < theta.size(); ++k) { 
        if (std::abs(theta(k)) < epsilon) {
            small_coeff_count++;
        }
    }

    if (log_count % 100 == 0){
        std::ofstream file(filename, std::ios::app); 
        if (file.is_open()) {
            if (file.tellp() == 0) {
                file << "RMSE,Penalty\n";  
            }
            file << rmse << "," << selected_terms_local.size() << "\n";  
            file.close();
        } else {
            std::cerr << "Unable to open file to write fitness data.\n";
        }
    }

    log_count++;

    penalty += coeff_lambda * small_coeff_count;

    return rmse + penalty;
}


std::vector<int> GeneticAlgorithm::tournament_selection(const std::vector<double>& fitnesses) {
    std::uniform_int_distribution<int> index_dist(0, population.size() - 1);
    std::vector<std::pair<std::vector<int>, double>> participants;

    for (int i = 0; i < tournament_size; ++i) {
        int idx = index_dist(rng);
        participants.push_back({ population[idx], fitnesses[idx] });
    }

     std::sort(participants.begin(), participants.end(), [](const std::pair<std::vector<int>, double>& a, const std::pair<std::vector<int>, double>& b) {
        return a.second < b.second;
    });

    return participants[0].first;   
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
    // Map genome indices to monomial exponents (i, j)
    selected_terms.clear();
    for (int idx = 0; idx < best_genome.size(); ++idx) {
        if (best_genome[idx] == 1) {
            int i = idx / (m + 1);
            int j = idx % (m + 1);
            selected_terms.emplace_back(i, j);
        }
    }

     Eigen::MatrixXd X_poly(X.rows(), selected_terms.size());
    for (int sample = 0; sample < X.rows(); ++sample) {
        for (size_t idx = 0; idx < selected_terms.size(); ++idx) {
            int i = selected_terms[idx].first;
            int j = selected_terms[idx].second;
            X_poly(sample, idx) = std::pow(X(sample, 0), i) * std::pow(X(sample, 1), j);
        }
    }

    Eigen::MatrixXd X_poly_augmented = Eigen::MatrixXd::Ones(X_poly.rows(), X_poly.cols() + 1);
    X_poly_augmented.block(0, 1, X_poly.rows(), X_poly.cols()) = X_poly;

    Eigen::VectorXd theta = (X_poly_augmented.transpose() * X_poly_augmented).ldlt().solve(X_poly_augmented.transpose() * Z);

    coeffs = theta.segment(1, theta.size() - 1);
    intercept = theta(0);
}
