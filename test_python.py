import numpy as np
import yaml
import genetic_algorithm as ga

def main(config_path):
    # Load configuration from YAML
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    data_size = config["data_size"]
    n = config["n"]
    m = config["m"]
    population_size = config["population_size"]
    generations = config["generations"]
    mutation_rate = config["mutation_rate"]
    tournament_size = config["tournament_size"]
    rng_seed = config["rng_seed"]
    patience = config["patience"]
    rmse_lambda = config["rmse_lambda"]
    coeff_lambda = config["coeff_lambda"]

    # Set random seed
    np.random.seed(rng_seed)

    # Generate synthetic data
    X_data = np.random.uniform(-10.0, 10.0, size=(data_size, 2))
    Z_data = (
        -0.2 * np.power(X_data[:, 0], 3) * np.power(X_data[:, 1], 5)
        + 0.3 * np.power(X_data[:, 0], 2) * X_data[:, 1]
        - 0.4 * X_data[:, 0] * np.power(X_data[:, 1], 2)
        + 0.5 * X_data[:, 0] * X_data[:, 1]
        + 1.0
    )

    # Create and run the GeneticAlgorithm
    genetic_algo = ga.GeneticAlgorithm(
        X_data, Z_data, n, m, population_size, generations, mutation_rate,
        tournament_size, patience, coeff_lambda, rmse_lambda
    )
    
    genetic_algo.setFileLogPath("/home/hesam/Desktop/masters/second_year/space photo/codes/GA/logs/log.csv")
    genetic_algo.run()

    # Print results
    genetic_algo.print_results()
    coefficients = genetic_algo.get_coefficients()
    intercept = genetic_algo.get_intercept()
    selected_terms = genetic_algo.get_selected_terms()

    print("\nFinal Results:")
    print("Coefficients:", coefficients)
    print("Intercept:", intercept)
    print("Selected Terms (x^i y^j):", selected_terms)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Please provide the path to the YAML configuration file.")
        sys.exit(-1)
    config_path = sys.argv[1]
    main(config_path)
