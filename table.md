
Table 1: Grid convergence study over 3 grids. phi represents the {INSERT MEANING OF PHI HERE} and phi_extrapolated its extrapolated value. N_cells is the number of grid elements, r the refinement ration between two successive grids. GCI is the grid convergence index in percent and its asymptotic value is provided by GCI_asymptotic, where a value close to unity indicates a grid independent solution. The order achieved in the simulation is given by p.

|        |  phi      |   N_cells   |  r  |  GCI  | GCI_asymptotic |  p   | phi_extrapolated |
|--------|:---------:|:-----------:|:---:|:-----:|:--------------:|:----:|:----------------:|
|        |           |             |     |       |                |      |                  |
| Grid 1 | 6.063e+00 |       18000 | 1.3 | 2.17% |                |      |                  |
| Grid 2 | 5.972e+00 |        8000 | 1.2 | 4.11% |      1.015     | 2.30 |     6.17e+00     |
| Grid 3 | 5.863e+00 |        4500 | -   | -     |                |      |                  |
|        |           |             |     |       |                |      |                  |
Generated using pyGCS (Grid Convergence Study)
- https://github.com/tomrobin-teschner/pyGCS
- https://pypi.org/project/pygcs/