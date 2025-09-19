# Optimization Methods for the Household Assignment Problem

This repository contains implementations of the optimization methods developed and applied in the context of my doctoral research.
The methods are organized in separate folders, and the script `hap_solving.py` is provided at the root level to run selected methods.


## Installation

The methods are implemented in `Python 3.11.5` and the decomposition approach uses `Gurobi 10.0.3`.
Academic licenses are available at [Gurobi's website](https://www.gurobi.com/academia/academic-program-and-licenses).

Clone the repository and install the remaining required dependencies:

```bash
git clone https://github.com/lucas-moschen/hap-optimization-methods.git
cd hap-optimization-methods
pip install -r requirements.txt
```


## Usage

### Generating the assignment weight matrix

First, generate the assignment weight matrix according to Section 5.3 of the thesis by running

```bash
python3 w_creation.py w_type "data_words" tau
```

at the root level.

### Arguments:

`w_type` Type of the matrix of assignment weights.

`data_words` List of sub-strings used to find the data set files.

`tau` The vector of coefficients to compute the assignment weights according to Section 5.3.2 of the thesis. 
It is necessary only if `w_type == "w_2"`. 

### Example:

```bash
python3 w_creation.py w_2 "['(9700hhd)']" [0.4,0.4,0.2]
```

### Running the solution methods

After constructing the matrix, run the solution methods with

```bash
python3 hap_solving.py "data_words" w_name n vartheta_hhd vartheta_per method params
```

at the root level.

### Arguments:

`data_words` List of sub-strings used to find the data set files.

`w_name` Name of the file containing the matrix of assignment weights.

`n` Number of constrained grid cells for each type of side constraints, which are selected according to the criterion described in Section 8.2 of the thesis.

`vartheta_hhd` 𝜗 value applied in the constraints on the number of households allocated in grid cells, as described in Section 8.2 of the thesis.

`vartheta_per` 𝜗 value applied in the constraints on the number of people allocated in grid cells, as described in Section 8.2 of the thesis.

`method` Solution method selected.

`params` Parameter values of the corresponding solution method. This can be empty (no parameters), a single value, or multiple values depending on the selected method.

### Examples:

```bash
python3 hap_solving.py "['(9700hhd)']" "w_2_city10000(9700hhd)seed=10[0.4, 0.4, 0.2].csv" 46 0.95 0.95 exact
```
runs the exact solution, 
```bash
python3 hap_solving.py "['(9700hhd)']" "w_2_city10000(9700hhd)seed=10[0.4, 0.4, 0.2].csv" 46 0.95 0.95 greedy
```
runs the greedy heuristic, and
```bash
python3 hap_solving.py "['(9700hhd)']" "w_2_city10000(9700hhd)seed=10[0.4, 0.4, 0.2].csv" 46 0.95 0.95 path_gr
```
run the path-growing heuristic. These three methods have no parameters.

The regional decomposition with the default parameters obtained in Section 8.3 of the thesis can be run with
```bash
python3 hap_solving.py "['(9700hhd)']" "w_2_city10000(9700hhd)seed=10[0.4, 0.4, 0.2].csv" 46 0.95 0.95 reg_decomp 5000 0.97 0.05 0.3 100
```

Analogously, the decomposition by household size can be run with 
```bash
python3 hap_solving.py "['(9700hhd)']" "w_2_city10000(9700hhd)seed=10[0.4, 0.4, 0.2].csv" 46 0.95 0.95 hhd_decomp 2500 0.97 0.05 0.3
```
and the LRBAM-SPP with 
```bash
python3 hap_solving.py "['(9700hhd)']" "w_2_city10000(9700hhd)seed=10[0.4, 0.4, 0.2].csv" 46 0.95 0.95 lrbam_spp 0.075 0.01 1.01
```
The LRBAM-IWPP can be executed simply by replacing the argument `lrbam_spp` with `lrbam_iwpp` in the previous command.


## Repository Structure
```bash
hap-optimization-methods/
│
├── common/			# Shared utility modules
├── data/			# Folder that stores inputs and outputs 
├── decomposition/		# Decomposition approach
├── greedy_heuristic/		# Greedy heuristic
├── lrbam/			# Lagrangian-relaxation-based approximation method
├── path_growing_heuristic/	# Path-growing heuristic
│
├── w_creation.py		# Script to create assignment weight matrices
├── hap_solving.py		# Root script to run selected methods
├── requirements.txt		# Dependencies
├── LICENSE			# License file (GPL-3.0)
├── CITATION.cff		# Citation file
├── AUTHORS			# List of contributors
└── README.md			# This file
```


## Citation
If you use this code, please cite the related article and thesis:
See `CITATION.cff` file.


## License
This repository is licensed under the GPL-3.0 License.


## Author
Lucas Moschen

Doctoral Researcher, University of Trier

See the `AUTHORS` file for a complete list of contributors.


## Acknowledgments

Parts of the `common/get_files.py` module in this project build upon code from [msc-thesis-microsimulations](https://github.com/ReiterKM/msc-thesis-microsimulations) by Kendra M. Reiter, licensed under the GPL-3.0.
