import argparse
import json
import os
import numpy as np
from fidelity_quantum_kernel_efficient import FidelityQuantumKernel

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, SamplerV2
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager   
from qiskit_machine_learning.state_fidelities import ComputeUncompute   





ap=argparse.ArgumentParser()
ap.add_argument('-config','--config_file',
                default='Code/config_test.json',
                required=False,
                help='json file with experiments info path')

args=vars(ap.parse_args())
params_dir=args['config_file']
############################PARAMETERS#########################################################
###########Load hyperparameters from json################
print('Loading Parameters')
#load the json file
with open(params_dir) as json_file:
    params = json.load(json_file)

output_path=params['output_path']

################################ connect_to_backend_And_simulator #########################################


service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='a0eaa4d6a98938f304126de89ec0a6c241b5b12801420f71256d1966bad2d28df4b39e8e9bdc2bd63b5a2c4691318743f17bb269d9fad11db1635f0614e206d8'
)



#connection to a real backend
backend_QPU = service.backend(params['backend_name'])#.least_busy(simulator=False, operational=True)  # to select the least busy quantum machine available
num_qubits_backend_QPU = backend_QPU.num_qubits
session_QPU = Session(backend=backend_QPU)

sampler_QPU = SamplerV2(mode=session_QPU)

#connection to a simulator
num_qubits_backend_sim = 4
backend_sim = GenericBackendV2(num_qubits=num_qubits_backend_sim, seed = 1)                         
session_sim = Session(backend=backend_sim)
sampler_sim = SamplerV2(mode=session_sim)


################################# start the experiment #########################################

for number_of_qubits in params['number_of_qubits']:


################################# create data #########################################
    np.random.seed(seed=params['seed'])
    X = np.random.uniform(-np.pi/2, np.pi/2, size = (params['number_of_points_X'],number_of_qubits))
    Y = np.random.uniform(-np.pi/2, np.pi/2, size = (params['number_of_points_X'],number_of_qubits))


    sub_path = f'{output_path}{number_of_qubits}_qubits/'
    # Create the directory if it does not exist 
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

################################## create the pass manager and the feature map ######################################### 
    #define a feature map 
    feature_map = ZZFeatureMap(feature_dimension=number_of_qubits, reps=1, entanglement='linear')
    
    for optimization_level  in params['optimization_levels']:
        #create the pass manager
        pass_manager_QPU = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend_QPU)
        pass_manager_sim = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend_sim)

        sub_sub_path = f'{sub_path}opt_{optimization_level}/'
        # Create the directory if it does not exist
        if not os.path.exists(sub_sub_path):
            os.makedirs(sub_sub_path)
################################## compute kernels ##########################################


################### SIMULATED MATRIX ##############################
        #compute the simulated kernel matrix
        pass_manager_sim = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend_sim)
        fidelity_sim = ComputeUncompute(sampler=sampler_sim,  pass_manager=pass_manager_sim)
        qkernel_sim = FidelityQuantumKernel(feature_map = feature_map, fidelity = fidelity_sim)
        kernel_matrix_sim = qkernel_sim.evaluate(X, Y)
        #write the simulated matrix to file
        name = f'kernel_matrix_simulated_{number_of_qubits}_qubits_opt_{optimization_level}.txt'
        np.savetxt(sub_sub_path + name, kernel_matrix_sim)

        #load the simulated matrix from file
        #kernel_matrix_sim = np.loadtxt(sub_sub_path + name)

        
################### STANDARD IMPLEMENTATION ##############################
        #compute the kernel matrix on a QPU with the standard implementation
        fidelity_QPU = ComputeUncompute(sampler=sampler_QPU,  pass_manager=pass_manager_QPU)
        qkernel_QPU = FidelityQuantumKernel(feature_map = feature_map, fidelity = fidelity_QPU)
        kernel_matrix_QPU = qkernel_QPU.evaluate(X, Y)

        #write the kernel matrix and the metrics to file
        name = f'kernel_matrix_QPU_standard_implementation_{number_of_qubits}_qubits_opt_{optimization_level}.txt'
        np.savetxt(sub_sub_path + name, kernel_matrix_QPU)


        #compute the average error and the qpu time
        avg_error = np.mean(np.abs(kernel_matrix_QPU - kernel_matrix_sim))
        quanutm_seconds = 0  #numbers are taken from IBM dashboard directly and written to file 


       
        metrics = {'quantum_seconds': quanutm_seconds, 
                   'average_error': avg_error}
        #write the metrics to file
        name = f'QPU_standard_implementation_{number_of_qubits}_qubits_opt_{optimization_level}_metrics.txt'
        with open(sub_sub_path + name, 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
        
################### EFFICIENT IMPLEMENTATION ##############################

        shots = params['number_of_shots']
        for density in params['density']:
            fidelity_efficient_QPU = ComputeUncompute(sampler=sampler_QPU)
            qkernel_efficient_QPU = FidelityQuantumKernel(feature_map = feature_map, fidelity = fidelity_efficient_QPU)
            kernel_matrix_efficient_QPU = qkernel_efficient_QPU.evaluate_efficient(X, y_vec= Y,  backend = backend_QPU, pass_manager = pass_manager_QPU, shots = shots, density= density)

            #compute average error with respect to the simulated matrix and the qpu time
            avg_error = np.mean(np.abs(kernel_matrix_efficient_QPU - kernel_matrix_sim))
            quanutm_seconds = qkernel_efficient_QPU._sampler_job_extended.metrics()['usage']['quantum_seconds']
            metrics = {'quantum_seconds': quanutm_seconds,
                       'average_error': avg_error}
            
            #write the metrics to file 
            name = f'QPU_efficient_implementation_{number_of_qubits}_qubits_opt_{optimization_level}_{density}_density_metrics.txt'
            with open(sub_sub_path + name, 'w') as f:   
                for key, value in metrics.items():
                    f.write(f'{key}: {value}\n')
            #write the kernel matrix to file
            name = f'kernel_matrix_efficient_QPU_{number_of_qubits}_qubits_opt_{optimization_level}_{density}_density.txt'
            np.savetxt(sub_sub_path + name, kernel_matrix_efficient_QPU)
            


































# Create the directory if it does not exist
#if not os.path.exists(output_path):
#    os.makedirs(output_path)
#    print(f"Directory {output_path} created")
#else:
#    print(f"Directory {output_path} already exists")
