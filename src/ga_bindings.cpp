#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include "GA.hpp"
#include <pybind11/stl.h>  

namespace py = pybind11;

PYBIND11_MODULE(genetic_algorithm, m) {
    py::class_<GeneticAlgorithm>(m, "GeneticAlgorithm")
        .def(py::init<const Eigen::MatrixXd&, const Eigen::VectorXd&, int, int, int, int, double, int, int, double, double>(),
             py::arg("X"), py::arg("Z"), py::arg("n"), py::arg("m"),
             py::arg("pop_size"), py::arg("generations"), py::arg("mutation_rate"),
             py::arg("tournament_size"), py::arg("patience"), py::arg("coeff_lambda"), py::arg("rmse_lambda"))
        .def("run", &GeneticAlgorithm::run)
        .def("setFileLogPath", &GeneticAlgorithm::setFileLogPath, py::arg("filename"))
        .def("print_results", &GeneticAlgorithm::print_results)
        .def("get_coefficients", &GeneticAlgorithm::getCoefficients)
        .def("get_intercept", &GeneticAlgorithm::getIntercept)
        .def("get_best_genome", &GeneticAlgorithm::getBestGenome)
        .def("get_selected_terms", &GeneticAlgorithm::getSelectedTerms);
}
