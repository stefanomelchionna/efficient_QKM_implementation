# Enhancing Efficiency in QKM via Parallel Circuit Implementation.

# PROJECT DESCRIPTION

In recent years, Quantum Kernel Methods (QKMs) have emerged as promising tools for a range of machine learning tasks, including classification, clustering, and anomaly or novelty detection. Superior performance of QKMs with respect to classical kernel methods has been proven on a variety of datasets.
The popularity of these methods is also due to open-source implementations with a high degree of abstraction, such as Qiskit Machine Learning (https://qiskit-community.github.io/qiskit-machine-learning/).
However, the practical implementation of QKMs is accompanied by several challenges. As with all kernel methods, QKMs exhibit quadratic scaling with the size of the dataset: to compute a kernel for a dataset of n points, one needs to run n(n−1)/2 circuits (each with several shots). In the current implementation, each circuit is submitted to the QPU as an independent job.
The number of qubits required by each circuit is generally equal (or proportional) to the number of features in the dataset, which is a small number in many applications. Therefore, in cases where the number of qubits required is smaller than the number of qubits available on a QPU, the current implementation of QKMs is inefficient as it underutilizes the number of qubits available.
With this project, we aim to increase efficiency by implementing several circuits in parallel. In this way, we can 1) make efficient use of all the qubits on a given QPU without increasing the circuits' depth, and 2) reduce the number of jobs required to calculate a given kernel.

The image below contrasts the current implementation and our proposed implementation for calculating a kernel derived from a dataset of 100 points on a 156-qubit machine. The image shows the number of jobs required to calculate the kernel (on the vertical axis) when varying the dataset dimensionality (here we assume the number of features in the dataset equals the number of qubits used for calculating each entry of the kernel - this is the typical case). The graph shows a decisive advantage of our method for datasets with low dimensionality compared to the total number of qubits available (156 in our example). 


![Scaling](Images/Scaling_image.png)



Note that a lower number of jobs submitted translates into a low QPU time utilization and therefore to significant cost saving. 
Our implementation will be open source and compatible with the widely used Qiskit Machine Learning 0.8.2 library. With our work we will contribute directly to the qiskit-machine-learning project. 

 # CURRENT IMPLEMENTATION

The current implementation available in this GitHub is a prototype, which only supports symmetric fidelity-kernels which are small enough to be computed with a single job. 
The goal of this prototype is only to show that the method can work and to run a first test on real QPUs. 
The goal of the project is to create a universal implementation supporting kernels of all types and sizes.

# PRELIMINARY RESULTS
We tested our prototype on a IBM 127 qubits machine (ibm_sherbrooke) to compute a 10x10 symmetric kernel where each entry is computed by a 2 qubits circuit. The default wisket-machine-learning implementation required the run of 45 jobs and 50 seconds of QPU time. 
Our implementation required a single job and 2 seconds of QPU time. On this particular task we were therefore 25x faster. 
