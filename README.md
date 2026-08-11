# FastSlipPy: Induced Seismic Fault Slip Modeling

<p align=center><img height="80.0%" width="80.0%" src="https://raw.githubusercontent.com/ChengshunShang1996/FastSlipPy/master/docs/images/logo.png"></p>

![Release][release-image]
![License][license-image]
![Contributing][contributing-image]

<!---
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16282499.svg)](https://doi.org/10.5281/zenodo.16282499)
-->

FastSlipPy is a scientific software package for modeling dynamic fault slip in induced seismicity. It is designed to simulate the behavior of faults under various stress conditions, providing insights into the mechanisms driving induced earthquakes. The software is based on the Finite Difference Method (FDM).

The initial version of this software is based on the open-source code [IndNuc][IndNuc_link], which is a Matlab-based code specifically developed for modeling induced seismicity in Groningen area. FastSlipPy has been rewritten in Python to enhance its accessibility and usability, allowing for easier integration with other scientific tools and libraries.

The ambitious goal of FastSlipPy is to provide an easy-to-use, efficient, and flexible platform for the multi-scale and multi-physics modeling of induced seismicity, which may include the following features in the future:

- Multiscale modeling: coupling with Discrete Element Method (DEM) for the simulation of fault gouge.
- Multiphysics modeling: supporting the coupling of fluid, thermal, and mechanical processes.

## Table of Contents

- [Main Features](#main-features)
- [Dependencies](#demgen-dependencies)
- [Instructions](#instructions)
  - [Input and Output Files](#input-and-output-files)
  - [Running Simulations](#running-simulations)
  - [Checking Results](#checking-results)
- [Examples](#examples)
- [Documentation](#documentation)
- [How to Contribute](#how-to-contribute)
- [How to Cite](#how-to-cite)
- [Authorship](#authorship)
- [License](#license)

## Main Features

This program can be used for modeling induced seismicity in a fault system. The main features of this program include:

- Easy-to-install and run on different platforms (Windows or Linux).
- Adaptive time stepping for efficient simulation of dynamic fault slip.
- Support different friction models, such as rate-and-state friction and others (under development).
- Support different fault angles.
- Post-processing in the format of both Paraview and Matplotlib for visualization of results.
- Modular design for easy integration with other scientific tools and libraries.

## Dependencies

FastSlipPy is fully written in the [Python][python_website] programming language and adopts the Object Oriented Programming (OOP) paradigm to offer modularity and extensibility.

Please make sure you have installed Python3.X.X on your PC. Currently, [Python3.10.X][python310_website] is recommended, as other versions haven't been tested.

All the required Python libraries will be added automatically. Some of them are:

- numpy
- matplotlib
- meshio

## Instructions

### Install

#### Install for users

> pip install fastslippy

#### Install for developers

Step 1: Download this software or use Git:

> git clone https://github.com/ChengshunShang1996/FastSlipPy.git

Step 2: Installation.

(Windows users can jump this step) For Linux users, the virtual environment is suggested to be used:

> python3 -m venv ~/my_env

> source ~/my_env/bin/activate

In the project folder, run cmd command:

> pip install -e .

#### Successful installation?

Test it:

> from fastslippy import FastSlipPy

For Linux or HPC users:

> python3 -c "from fastslippy import FastSlipPy; print('Success')"

If you see the logo of FastSlipPy, you have successfully installed it. To run an example, please check the [examples](#examples) section.

### Input and Output Files

* **Input Parameters**:

All the default parameters are defined in the file [model_parameters.py][model_parameters]. You can modify the parameters in the running script.

Grid stretching options are also available in [model_parameters.py][model_parameters]:

- `x_stretch_enabled`, `y_stretch_enabled`
- `x_stretch_inner_size`, `y_stretch_inner_size`
- `x_stretch_inner_points`, `y_stretch_inner_points`
- `x_stretch_power`, `y_stretch_power`
- `x_stretch_max_cell_size`, `y_stretch_max_cell_size` (optional caps for the largest stretched cells)

If a max-cell-size cap is violated, FastSlipPy raises a `ValueError` and reports a suggested larger odd `Nx`/`Ny` while keeping the inner mesh point count unchanged.

By default, FastSlipPy uses a uniform grid.

Stretched mesh support in the elastic solver is currently experimental. To run with stretched mesh anyway, set:

- `allow_nonuniform_solver=True`

If stretched mesh is enabled without this explicit opt-in, FastSlipPy now raises an error to prevent silently unreliable results.

For large-scale runs that hit memory limits during sparse LU factorization, you can switch to the iterative linear solver in [model_parameters.py][model_parameters]:

- `linear_solver="iterative"`
- `iterative_method="gmres"` (or `"bicgstab"`)
- `ilu_drop_tol`, `ilu_fill_factor`, `iterative_rtol`, `iterative_maxiter`

By default, `linear_solver="direct"` is kept for backward compatibility. If direct LU runs out of memory, FastSlipPy can automatically fall back to iterative mode with:

- `fallback_to_iterative_on_oom=True`

* **Output files**:

The output files are generated in the specified output directory [output] and can be visualized using Paraview or Matplotlib. The Matplotlib visualization is used by default, and the Paraview visualization can be enabled by setting the parameter `output_vtk_option` to `True`.

### Running Simulations

To run a simulation, you can use the provided example scripts in the [examples][examples_link] folder. As running other Python scripts, you can run the example script in the command line:

> python run_case_groningen.py

or with your preferred way to run Python scripts. The simulation will start, and the output files will be generated in the specified output directory.

## Examples

There are two examples in the [examples][examples_link] folder. Here are the example results:

* **Groningen case**

<p float="left">
<img src="https://raw.githubusercontent.com/ChengshunShang1996/FastSlipPy/master/docs/images/example_results_groningen_fields.png" height="500"/>
</p>

* **Lab-scale shear case**

<p float="left">
<img src="https://raw.githubusercontent.com/ChengshunShang1996/FastSlipPy/master/docs/images/example_results_lab_shear.png" height="500"/>
</p>

*A high Young's modulus is used in the lab-scale shear case to generate the above figure.*

## Documentation

Please read this README.md for information.

## How to Contribute

Please check the [contribution guidelines][contribute_link].

## How to Cite

To cite this repository, you can use the metadata from [this file][citation_link].

## Authorship

- **Chengshun Shang** <sup>1</sup> (<c.shang@uu.nl>)

<sup>1</sup> Utrecht University ([UU][uu_website])

<p float="left">
<img src="https://raw.githubusercontent.com/ChengshunShang1996/FastSlipPy/master/docs/images/uu.png" height="100"/>
</p>

## Acknowledgement

The program was initially developed under the context of the FastSlip project. The author would like to thank the project team (Dr. [André Niemeijer][andreniemeijer_link] et al.) for their support and guidance. The author also acknowledges the contributions of the open-source community, particularly the developer of [IndNuc][IndNuc_link], Dr. [Meng Li][mengli_link], whose work served as a foundation for this software.

## License

FastSlipPy is licensed under the [MIT license][bsd_license_link],
which allows the program to be freely used by anyone for modification, private use, commercial use, and distribution, only requiring the preservation of copyright and license notices.
No liability and warranty are provided.

[src_folder]: .src/
[IndNuc_link]: https://github.com/USustSub/IndNuc
[release-image]: https://img.shields.io/badge/release-0.1.1-green.svg?style=flat
[license-image]: https://img.shields.io/badge/license-MIT-green.svg?style=flat
[contributing-image]: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
[json_link]: https://www.json.org/
[contribute_link]: https://github.com/ChengshunShang1996/DEMGen/blob/main/CONTRIBUTING.md
[citation_link]: https://github.com/ChengshunShang1996/DEMGen/blob/main/CITATION.cff
[uu_website]: https://www.uu.nl/en
[bsd_license_link]: https://choosealicense.com/licenses/bsd-2-clause/
[python_website]: https://www.python.org/
[python310_website]: https://www.python.org/downloads/release/python-3100/
[examples_link]: ./examples/
[model_parameters]: ./src/fastslippy/pre_processing/model_parameters.py
[mengli_link]: https://github.com/limeng-uni
[andreniemeijer_link]: https://www.uu.nl/medewerkers/ARNiemeijer
