# Grid Converge Lab

Grid Convergence Index (GCI) Analysis is a technique used to assess the accuracy of numerical simulations, particularly those involving finite element methods (FEM), finite volume methods (FVM), or finite difference methods (FDM). In simple terms, GCI is a systematic method for quantifying the numerical error associated with mesh discretization in computational simulations.

## Importance of GCI

### Accuracy and Reliability
The primary function of GCI is to ensure that simulation results are accurate and reliable. By evaluating mesh convergence, engineers and scientists can determine if the mesh resolution is sufficient to capture the physical phenomena of interest. This is crucial to avoid making decisions based on inaccurate results.

### Resource Optimization
Simulations with very fine meshes can be extremely resource-intensive, leading to long processing times and high memory consumption. GCI helps find an appropriate balance between accuracy and computational cost, optimizing resource usage without compromising the quality of the results.

### Model Validation
GCI is a vital tool for validating numerical models. It provides a quantitative measure of discretization error, which is essential for comparing simulation results with experimental or theoretical data. This helps ensure that the model is correctly implemented and that its predictions are reliable.

### Risk Reduction
In many industrial applications, such as aerospace, automotive, and civil engineering, decisions based on numerical simulations have significant implications for safety and performance. GCI analysis helps mitigate risks associated with discretization errors, providing greater confidence in simulation results.

### Documentation and Reproducibility
Including GCI analysis in simulation reports enhances transparency and reproducibility of studies. Other researchers and engineers can better understand the limitations of the results and replicate the study with confidence in the presented conclusions.

In summary, Grid Convergence Index (GCI) Analysis is an essential practice in numerical simulations, providing a solid foundation for evaluating accuracy, optimizing resources, validating models, reducing risks, and improving scientific documentation.

## Development Team

- **Prof. Dr. Nicolas Spogis** - [nicolas.spogis@gmail.com](mailto:nicolas.spogis@gmail.com) | [Linktree](https://linktr.ee/CascaGrossaSuprema)
- **Prof. Dr. Diener Volpin Ribeiro Fontoura** - [dvolpin@gmail.com](mailto:dvolpin@gmail.com)

## Installation

To install the necessary dependencies, you need to have Python installed on your system. If you don't have Python, you can download it [here](https://www.python.org/downloads/). After installing Python, follow the steps below:

1. **Clone the Repository**: First, clone the GCI repository to your local machine.
2. **Install Dependencies**: Within the project directory, locate the `requirements.txt` file containing all necessary libraries. Install them by running:
   ```bash
   pip install -r requirements.txt