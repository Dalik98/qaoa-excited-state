# Excited states calculations with the quantum approximate optimization algorithm 
This is a concluded master thesis work at the University of Bologna, the thesis can be found [here](https://amslaurea.unibo.it/28792/).
## About the project
The `QAOA` is a quantum algorithm developed to solve combinatorial problem, such as the `Max-Cut`, finding the ground state of a Hamiltonian that encodes the solution
of the problem. 
The QAOA is not able to calculate the excited states of a Hamiltonian, in this work I will propose a procedure for finding the excited states.
The calculation of the first excited state is obtained giving an extra energy to the ground state, which is obtained through the transformation:

$$H_{\text {1° e.s.}}=H_{g.s.}+\lambda_{0}\ket{\psi_{\text {g.s.}}} \bra{\psi_{\text {g.s.}}}$$

Where $H_{g.s.}$ is the Hamiltonian we want to calculate the excited states, $\ket{\psi_{\text {g.s.}}}$ is the ground state we should calculate at priori, $\lambda_{0}$ is
a constant that should be greater than the energy gap between the ground state and the first excited state. It gives an extra energy to the ground state such that the
Hamiltonian $H_{\text {1° e.s. }}$ has for the ground state the excited state of $H_{g.s.}$ we are looking for. This new Hamiltonian can be used as cost Hamiltonian in the QAOA algorithm for the calculation of the first excited state.

The other excited states can be found applying the transformation recursively.
Transformation for the second excited state:

$$H_{\text {2° e.s.}}=H_{1° e.s.}+\lambda_{1}\ket{\psi_{\text {1° e.s.}}} \bra{\psi_{\text {1° e.s.}}}$$

To benchmark the procedure, the `LiH` (Lithium Hydride) Hamiltonian in second quantization has been used.

## Build with

The program is written in `Python 3.11`, using the following libraries:
* [QuTip](https://qutip.org/) Used for quantum mechanical calculations.
* [OpenFermion](https://quantumai.google/openfermion)  This library is used for the creation of the Lithium Hydride and Hydrogen molecule Hamiltonian.
* [Psi4](https://psicode.org/)  OpenFermion relies on this library for the calculation of the one-body, two-body and nuclear repulsion coefficients in the second quantized diatomic Hamiltonian.
* [SciPy](https://scipy.org/)  It is used for the implementation of the basin-hopping and BFGS optimization methods.
* [MatPlotLib](https://matplotlib.org/) It is used to generate plots.

## Usage

The program works with command line options, for example:
```
$ python main.py method basin -p 1to5 -t 10 -s 1 -a1 Li -a2 H
```
The command executes the qaoa classical simulation considering p values from 1 to 5, with 10 independent calculations, applied on the Lithium Hydride Hamiltonian. The basin-hopping optimization method is used for the optimization.

Where:
* `-p` specifies the p values to consider, `-p 1` considers only $p=1$, while `-p 1to5` considers all the values from 1 to 5.
* `-t` specifies how many indipendent runs for each value of p the qaoa simulation must be performed.
* `-s` specifies the state to calculate, `-s 0` means the ground state is calculated.
* `-a1 -a2` these options specify the atoms of the molecule to be considered.
* `method` specifies what operation perform, `method basin` executes the basin-hopping algorithms, `method diag` perform the calculation of the specified state using the diagonalization method instead of QAOA, `method average` performs the calculation of the average and dev. std. between all the independent calculation for a specific p value performed previously, `method corr` creates the plots that compare the calculation performed with different values of p.

## Results
The procedure we propose succeed to calculate the excited states, only when the states are not degenerate. The accuracy increases with p.

The following plot shows the ground state energy in function of the internuclear distance for the LiH molecule Hamiltonian. With $p=3$ we have already a good approximation.

![excited state calculation](https://github.com/danieletrisciani/qaoa-excited-states-calculation/assets/20107065/2ca59971-e431-4a6c-962d-4712d57da5b7)

To calculate the first excited state, we apply the trasformation previously introduced, to obtain the Hamiltonian $H_{\text {1° e.s.}}$. It will be the new cost Hamiltonian. 
The following plot shows the excited state energy in function of the internuclear distance for the LiH molecule Hamiltonian.


![ground state calculation](https://github.com/danieletrisciani/qaoa-excited-states-calculation/assets/20107065/aa173a89-717d-4bd2-abed-f29b88e41cfd)


