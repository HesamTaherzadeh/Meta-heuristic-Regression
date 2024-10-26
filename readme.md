![CI](https://github.com/HesamTaherzadeh/Meta-heuristic-Regression/actions/workflows/ci.yml/badge.svg)
# Genetic Algorithm for Polynomial Regression

This project implements a Genetic Algorithm (GA) to perform polynomial regression by selecting significant polynomial terms up to a specified degree. The GA optimizes the selection of polynomial degrees and their coefficients to fit a given dataset, minimizing the Root Mean Squared Error (RMSE) while penalizing model complexity.

## Features

- **Polynomial Degree Selection**: Automatically selects the most suitable polynomial degrees.
- **Coefficient Estimation**: Estimates coefficients for the selected degrees using linear regression.
- **Customizable GA Parameters**: Adjust population size, generations, mutation rate, and tournament size.
- **Test Suite**: Includes tests using Google Test to validate algorithm correctness.

---

## Dependencies

- C++11 or higher
- [Eigen 3.3+](http://eigen.tuxfamily.org/)
- [Google Test](https://github.com/google/googletest)

---