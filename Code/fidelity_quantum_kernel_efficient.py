# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Fidelity Quantum Kernel"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_machine_learning.state_fidelities import BaseStateFidelity
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.transpiler import PassManager
from qiskit.primitives import BitArray


from qiskit_ibm_runtime import IBMBackend

from qiskit_machine_learning.kernels.base_kernel import BaseKernel

KernelIndices = List[Tuple[int, int]]


class FidelityQuantumKernel(BaseKernel):
    r"""
    An implementation of the quantum kernel interface based on the
    :class:`~qiskit_machine_learning.state_fidelities.BaseStateFidelity` algorithm.

    Here, the kernel function is defined as the overlap of two quantum states defined by a
    parametrized quantum circuit (called feature map):

    .. math::

        K(x,y) = |\langle \phi(x) | \phi(y) \rangle|^2
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        fidelity: BaseStateFidelity | None = None,
        enforce_psd: bool = True,
        evaluate_duplicates: str = "off_diagonal",
        max_circuits_per_job: int = None,
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If ``None`` is given,
                :class:`~qiskit.circuit.library.ZZFeatureMap` is used with two qubits. If there's
                a mismatch in the number of qubits of the feature map and the number of features
                in the dataset, then the kernel will try to adjust the feature map to reflect the
                number of features.
            fidelity: An instance of the
                :class:`~qiskit_machine_learning.state_fidelities.BaseStateFidelity` primitive to be used
                to compute fidelity between states. Default is
                :class:`~qiskit_machine_learning.state_fidelities.ComputeUncompute` which is created on
                top of the reference sampler defined by :class:`~qiskit.primitives.Sampler`.
            enforce_psd: Project to the closest positive semidefinite matrix if ``x = y``.
                Default ``True``.
            evaluate_duplicates: Defines a strategy how kernel matrix elements are evaluated if
               duplicate samples are found. Possible values are:

                    - ``all`` means that all kernel matrix elements are evaluated, even the diagonal
                      ones when training. This may introduce additional noise in the matrix.
                    - ``off_diagonal`` when training the matrix diagonal is set to `1`, the rest
                      elements are fully evaluated, e.g., for two identical samples in the
                      dataset. When inferring, all elements are evaluated. This is the default
                      value.
                    - ``none`` when training the diagonal is set to `1` and if two identical samples
                      are found in the dataset the corresponding matrix element is set to `1`.
                      When inferring, matrix elements for identical samples are set to `1`.
            max_circuits_per_job: Maximum number of circuits per job for the backend. Please
               check the backend specifications. Use ``None`` for all entries per job. Default ``None``.
        Raises:
            ValueError: When unsupported value is passed to `evaluate_duplicates`.
        """
        super().__init__(feature_map=feature_map, enforce_psd=enforce_psd)

        eval_duplicates = evaluate_duplicates.lower()
        if eval_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Unsupported value passed as evaluate_duplicates: {evaluate_duplicates}"
            )
        self._evaluate_duplicates = eval_duplicates
        if fidelity is None:
            fidelity = ComputeUncompute(sampler=Sampler())
        self._fidelity = fidelity
        if max_circuits_per_job is not None:
            if max_circuits_per_job < 1:
                raise ValueError(
                    f"Unsupported value passed as max_circuits_per_job: {max_circuits_per_job}"
                )
        self.max_circuits_per_job = max_circuits_per_job


    #def _get_circuits(self, x_vec: np.ndarray, backend: IBMBackend, pass_manager: PassManager, shots = 1024):
        #This function was used for testing only and is not used in the final implementation
        #self._num_qubits_fidelity_circuit, self._num_circuits_per_job, self._num_kernel_entries = self._determine_efficient_job_specifications(x_vec, backend)
        #extended_fidelity_circuit, parameter_values_extended_fidelity_circuit, fidelity_circuit, extended_circuit = self._create_extended_fidelity_circuit(x_vec, pass_manager)
        #return fidelity_circuit, extended_circuit


    def evaluate_efficient(self, x_vec: np.ndarray, backend: IBMBackend, pass_manager: PassManager, shots = 1024) -> np.ndarray:
        '''
        Efficient implementation on real backend. Only the simmetric case is implemented.
        Only the Compute Uncompute fidelity is supported. 
        Only kernels which size can be run in a single jos are supported for now. 

        Params:
        x_vec: np.ndarray
            The input data to be used to calculate the kernel matrix
        backend: IBMBackend
            The backend to be used to run the circuits
        pass_manager: PassManager
            The pass manager to be used to transpile the circuits. Note that pass manager and backend information are nedeed here to
             adapt the run specifications to the specific backend and settings.
        Returns:
        np.ndarray
            The kernel matrix calculated using the fidelity algorithm
        '''

        # determine the number of qubits neede to compute each fidelity circuit, 
        # the number of circuits that can fit in parallel on a backend per each job 
        # and the total number of kernel entries to be calculated
        self._num_qubits_fidelity_circuit, self._num_circuits_per_job, self._num_kernel_entries = self._determine_efficient_job_specifications(x_vec, backend)
        
        #create the extended fidelity circuit which defines a single job and get its parameter vector
        extended_fidelity_circuit, parameter_values_extended_fidelity_circuit = self._create_extended_fidelity_circuit(x_vec, pass_manager)
        

        #run the circuit
        sampler_job_extended = self._fidelity._sampler.run([(extended_fidelity_circuit, parameter_values_extended_fidelity_circuit)], shots = shots)


        #postprocess the results
        all_single_circuit_results = self._postprocess_sampler_job_results(sampler_job_extended)
        global_fidelities = self._get_global_fidelities(all_single_circuit_results, shots=shots)


        #assign the fidelities to the kernel matrix based on the correct indexes
        left_parameters, right_parameters, indices = self._get_symmetric_parameterization(x_vec) 

        kernel_shape = (x_vec.shape[0], x_vec.shape[0])
        kernel_matrix = np.ones((kernel_shape))

        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = global_fidelities[i]     
            kernel_matrix[row, col] = global_fidelities[i]      


        return kernel_matrix


    def _get_global_fidelities(self, all_single_circuit_results, shots):
        fidelities = []
        
        for single_circ_result in all_single_circuit_results:
            fidelity = self._get_global_fidelities_from_results_one_circuit(single_circ_result.astype(np.uint8), shots)
            fidelities.append(fidelity)
        return fidelities

    def _get_global_fidelities_from_results_one_circuit(self, results_one_circuit, shots):

        #transform the results into BitArray type, get the counts and the probability distribution
        results_one_circuit_bit_array = BitArray(results_one_circuit, self._num_qubits_fidelity_circuit)
        bitstring_counts = results_one_circuit_bit_array.get_counts()
        probability_distribution = {k: v / shots for k, v in bitstring_counts.items()}

        fidelity = probability_distribution.get("0"*self._num_qubits_fidelity_circuit, 0)  
        
        return fidelity



    def _postprocess_sampler_job_results(self, sampler_job_extended):
        
        #get the bitstrings
        bitstrings = sampler_job_extended.result()[0].data.c.get_bitstrings()

        #create a matrix from the bitstrings by breaking each depending on the number of qubits used by each fidelity circuit 
        bitstring_matrix = []
        for item in bitstrings:
            groups = [item[i:i+self._num_qubits_fidelity_circuit] for i in range(0, len(item), self._num_qubits_fidelity_circuit)] 
            bitstring_matrix.append(groups) 
        bitstring_matrix = np.array(bitstring_matrix)

        #Starting from the matrix collecting all the results we create the list of individual circuits results
        all_single_circuit_results = [bitstring_matrix[:,[-k-1]] for k in range(bitstring_matrix.shape[1])] #TODO: understand why bitstrings are returned flipped

        return all_single_circuit_results


    
    def _create_extended_fidelity_circuit(self, x_vec: np.ndarray, pass_manager):
        
        #start from the fidelity circuit and 
        fidelity_circuit = self.fidelity._construct_circuits([self._feature_map], [self._feature_map])[0]

        #get the parametrization of all the fidelity circuits which need to be run to compute the full kernel matrix
        left_parameters, right_parameters, indices = self._get_symmetric_parameterization(x_vec)
        values = self._fidelity._construct_value_list([self._feature_map]*self._num_kernel_entries, [self._feature_map]*self._num_kernel_entries, left_parameters, right_parameters)  #parameters values given to the sampler
        
        #loop over the number of circuits to be run in parallel and create the extended circuit by justapposing 
        # the individual fidelity circuits with the correct parametrization
        extended_circuit = QuantumCircuit(self._num_circuits_per_job*self._num_qubits_fidelity_circuit, self._num_circuits_per_job*self._num_qubits_fidelity_circuit)
        for i in range(self._num_circuits_per_job):

            params_fidelity_circuit = list(fidelity_circuit.parameters)
            parameters_x_this_rep = ParameterVector(f"rep_{i}_x", fidelity_circuit.num_parameters//2)
            parameters_y_this_rep = ParameterVector(f"rep_{i}_y", fidelity_circuit.num_parameters//2)
            parameters_this_rep = list(parameters_x_this_rep) + list(parameters_y_this_rep)
            param_map = {params_fidelity_circuit[i]: parameters_this_rep[i] for i in range(len(params_fidelity_circuit))}

            qubits = [i*self._num_qubits_fidelity_circuit +j for j in range(self._num_qubits_fidelity_circuit)]
            
            re_parametrized_fidelity_circuit = fidelity_circuit.assign_parameters(param_map)
            extended_circuit.compose(re_parametrized_fidelity_circuit, qubits=qubits, clbits=qubits, inplace=True)

        #transpile the circuit and assign the parameter values to the parameters
        extended_circuit_transpliled = pass_manager.run(extended_circuit)
        values_extended_circ = list(np.array(values).flatten())     
        


        return extended_circuit_transpliled, values_extended_circ



    def _determine_efficient_job_specifications(self, x_vec: np.ndarray, backend: IBMBackend):

        num_qubits_backend = backend.num_qubits                    #total number of qubits available on the backend
        num_qubits_fidelity_circuit = self.feature_map.num_qubits  #number of qubits of each small individual circuit defined by the feature map
        num_kernel_entries = x_vec.shape[0] * (x_vec.shape[0] -1) // 2 #number of distinct kernel entries to be calculated 
        num_circuits_per_job = min(int(np.floor(num_qubits_backend / num_qubits_fidelity_circuit)), num_kernel_entries)  #number of fidelity circuits parallelized on the backend for each job
        


        return num_qubits_fidelity_circuit, num_circuits_per_job, num_kernel_entries



    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:
        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        # determine if calculating self inner product
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        kernel_shape = (x_vec.shape[0], y_vec.shape[0])

        if is_symmetric:
            left_parameters, right_parameters, indices = self._get_symmetric_parameterization(x_vec)
            kernel_matrix = self._get_symmetric_kernel_matrix(
                kernel_shape, left_parameters, right_parameters, indices
            )
        else:
            left_parameters, right_parameters, indices = self._get_parameterization(x_vec, y_vec)
            kernel_matrix = self._get_kernel_matrix(
                kernel_shape, left_parameters, right_parameters, indices
            )

        if is_symmetric and self._enforce_psd:
            kernel_matrix = self._make_psd(kernel_matrix)

        return kernel_matrix

    def _get_parameterization(
        self, x_vec: np.ndarray, y_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, KernelIndices]:
        """
        Combines x_vec and y_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = x_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = np.asarray(
            [
                (i, j)
                for i, x_i in enumerate(x_vec)
                for j, y_j in enumerate(y_vec)
                if not self._is_trivial(i, j, x_i, y_j, False)
            ]
        )

        if indices.size > 0:
            left_parameters = x_vec[indices[:, 0]]
            right_parameters = y_vec[indices[:, 1]]

        return left_parameters, right_parameters, indices.tolist()

    def _get_symmetric_parameterization(
        self, x_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, KernelIndices]:
        """
        Combines two copies of x_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = x_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = np.asarray(
            [
                (i, i + j)
                for i, x_i in enumerate(x_vec)
                for j, x_j in enumerate(x_vec[i:])
                if not self._is_trivial(i, i + j, x_i, x_j, True)
            ]
        )

        if indices.size > 0:
            left_parameters = x_vec[indices[:, 0]]
            right_parameters = x_vec[indices[:, 1]]

        return left_parameters, right_parameters, indices.tolist()

    def _get_kernel_matrix(
        self,
        kernel_shape: tuple[int, int],
        left_parameters: np.ndarray,
        right_parameters: np.ndarray,
        indices: KernelIndices,
    ) -> np.ndarray:
        """
        Given a parameterization, this computes the symmetric kernel matrix.
        """
        kernel_entries = self._get_kernel_entries(left_parameters, right_parameters)

        # fill in trivial entries and then update with fidelity values
        kernel_matrix = np.ones(kernel_shape)

        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = kernel_entries[i]

        return kernel_matrix

    def _get_symmetric_kernel_matrix(
        self,
        kernel_shape: tuple[int, int],
        left_parameters: np.ndarray,
        right_parameters: np.ndarray,
        indices: KernelIndices,
    ) -> np.ndarray:
        """
        Given a set of parameterization, this computes the kernel matrix.
        """
        kernel_entries = self._get_kernel_entries(left_parameters, right_parameters)
        kernel_matrix = np.ones(kernel_shape)

        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = kernel_entries[i]
            kernel_matrix[row, col] = kernel_entries[i]

        return kernel_matrix

    def _get_kernel_entries(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray
    ) -> Sequence[float]:
        """
        Gets kernel entries by executing the underlying fidelity instance and getting the results
        back from the async job.
        """
        num_circuits = left_parameters.shape[0] 
        kernel_entries = []
        # Check if it is trivial case, only identical samples
        if num_circuits != 0:
            if self.max_circuits_per_job is None:
                job = self._fidelity.run(
                    [self._feature_map] * num_circuits,
                    [self._feature_map] * num_circuits,
                    left_parameters,  # type: ignore[arg-type]
                    right_parameters,  # type: ignore[arg-type]
                )
                kernel_entries = job.result().fidelities
            else:
                # Determine the number of chunks needed
                num_chunks = (
                    num_circuits + self.max_circuits_per_job - 1
                ) // self.max_circuits_per_job
                for i in range(num_chunks):
                    # Determine the range of indices for this chunk
                    start_idx = i * self.max_circuits_per_job
                    end_idx = min((i + 1) * self.max_circuits_per_job, num_circuits)
                    # Extract the parameters for this chunk
                    chunk_left_parameters = left_parameters[start_idx:end_idx]
                    chunk_right_parameters = right_parameters[start_idx:end_idx]
                    # Execute this chunk
                    job = self._fidelity.run(
                        [self._feature_map] * (end_idx - start_idx),
                        [self._feature_map] * (end_idx - start_idx),
                        chunk_left_parameters,  # type: ignore[arg-type]
                        chunk_right_parameters,  # type: ignore[arg-type]
                    )
                    # Extend the kernel_entries list with the results from this chunk
                    kernel_entries.extend(job.result().fidelities)
        return kernel_entries

    # pylint: disable=too-many-positional-arguments
    def _is_trivial(
        self, i: int, j: int, x_i: np.ndarray, y_j: np.ndarray, symmetric: bool
    ) -> bool:
        """
        Verifies if the kernel entry is trivial (to be set to `1.0`) or not.

        Args:
            i: row index of the entry in the kernel matrix.
            j: column index of the entry in the kernel matrix.
            x_i: a sample from the dataset that corresponds to the row in the kernel matrix.
            y_j: a sample from the dataset that corresponds to the column in the kernel matrix.
            symmetric: whether it is a symmetric case or not.

        Returns:
            `True` if the entry is trivial, `False` otherwise.
        """
        # if we evaluate all combinations, then it is non-trivial
        if self._evaluate_duplicates == "all":
            return False

        # if we are on the diagonal and we don't evaluate it, it is trivial
        if symmetric and i == j and self._evaluate_duplicates == "off_diagonal":
            return True

        # if don't evaluate any duplicates
        if np.array_equal(x_i, y_j) and self._evaluate_duplicates == "none":
            return True

        # otherwise evaluate
        return False

    @property
    def fidelity(self):
        """Returns the fidelity primitive used by this kernel."""
        return self._fidelity

    @property
    def evaluate_duplicates(self):
        """Returns the strategy used by this kernel to evaluate kernel matrix elements if duplicate
        samples are found."""
        return self._evaluate_duplicates
