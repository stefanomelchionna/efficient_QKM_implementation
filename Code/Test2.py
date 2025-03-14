from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

theta = Parameter('θ')
phi = Parameter('φ')

qc1 = QuantumCircuit(2)
qc1.h(0)
qc1.cx(0, 1)
qc1.rz(theta, 0)
qc1.ry(phi, 1)

print(qc1)

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

alpha = Parameter('α')
beta = Parameter('β')

qc2 = QuantumCircuit(3)
qc2.rx(alpha, 0)
qc2.ry(beta, 1)
qc2.cz(0, 2)
qc2.h(2)

print(qc2)

# Define your parameter sets
params1 = [3.1415/4, 3.1415/3] 
params2 = [3.1415/2, 3.1415/4]
params3 = [3.1415/6, 3.1415/2]
params4 = [3.1415/3, 3.1415/6]




from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, SamplerV2
import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager 
from qiskit.providers.fake_provider import GenericBackendV2
# Initialize Qiskit Runtime Service
service = QiskitRuntimeService()

#connection to a simulator
num_qubits_backend_sim = 20
backend_sim = GenericBackendV2(num_qubits=num_qubits_backend_sim, seed = 1)                         
session_sim = Session(backend=backend_sim)
pass_manager_sim = generate_preset_pass_manager(optimization_level=1, backend=backend_sim)
sampler_sim = SamplerV2(mode=session_sim)

qc1_transpilled = pass_manager_sim.run(qc1)
qc2_transpilled = pass_manager_sim.run(qc2)
pub1 = (qc1_transpilled, [params1, params2])
pub2 = (qc2_transpilled, [params3, params4])

job = sampler_sim.run([pub1, pub2], shots = 1)
result = job.result()

# Print the result
print(result.get_counts(pub1))