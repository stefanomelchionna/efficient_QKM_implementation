from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, SamplerV2
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager   
from qiskit_machine_learning.state_fidelities import ComputeUncompute    
from fidelity_quantum_kernel_efficient import FidelityQuantumKernel 


num_features = 2        #this is the dimensionality of every point in the dataset 
#we map every feature to a qubit in the fidelity quantum circuit used to compute each kernel entry. 
#Therefore, num_features is also the number of qubits used by the fidelity circuit.
number_of_points = 4


#create data
np.random.seed(seed=123)
X = np.random.uniform(-np.pi/2, np.pi/2, size = (number_of_points,num_features))


service = QiskitRuntimeService()

#connection to a real backend
backend_QPU = service.least_busy(simulator=False, operational=True)  # to select the least busy quantum machine available
num_qubits_backend_QPU = backend_QPU.num_qubits
session_QPU = Session(backend=backend_QPU)
pass_manager_QPU = generate_preset_pass_manager(optimization_level=1, backend=backend_QPU)
sampler_QPU = SamplerV2(mode=session_QPU)

#connection to a simulator
num_qubits_backend_sim = 10
backend_sim = GenericBackendV2(num_qubits=num_qubits_backend_sim, seed = 1)                         
session_sim = Session(backend=backend_sim)
pass_manager_sim = generate_preset_pass_manager(optimization_level=1, backend=backend_sim)
sampler_sim = SamplerV2(mode=session_sim)


#define a feature map 
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1, entanglement='linear')

fidelity_efficient_sim = ComputeUncompute(sampler=sampler_sim)
qkernel_efficient_sim = FidelityQuantumKernel(feature_map = feature_map, fidelity = fidelity_efficient_sim)
kernel_matrix_efficient = qkernel_efficient_sim.evaluate_efficient(X, backend_sim, pass_manager = pass_manager_sim, shots = 1024)

kernel_matrix = qkernel_efficient_sim.evaluate(X)





#fidelity_efficient_QPU = ComputeUncompute(sampler=sampler_QPU)
#qkernel_efficient_QPU = FidelityQuantumKernel(feature_map = feature_map, fidelity = fidelity_efficient_QPU)
#sampler_job_extended = qkernel_efficient_QPU.evaluate_efficient(X, backend_QPU, pass_manager = pass_manager_QPU, shots = 1024)


#print(len(pubs))
#print(num_qubits_backend_QPU)
#print(pubs[0][0])
#print(len(pubs[0][1]))
#print(len(pubs[1][1]))
#print(len(pubs[2][1]))
#print(pubs[-1][0])
#print(len(pubs[-1][1]))

#print(qkernel_efficient_QPU._num_kernel_entries)
#print(qkernel_efficient_QPU._num_jobs)


print(kernel_matrix)
print(kernel_matrix_efficient)