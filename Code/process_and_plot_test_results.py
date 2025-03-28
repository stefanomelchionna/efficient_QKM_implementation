import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

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



################################## POSTPROCESSING #########################################
# load all the metrics files and plot graphs


#create a scatter plot of quantum seconds vs average error for any given density and optimization level
for number_of_qubits in params['number_of_qubits']:
    for optimization_level in params['optimization_levels']:

        densities = []
        quantum_seconds = []
        average_errors = []
        for density in params['density']:
            densities.append(density)
            # load the metrics file
            path = f'{output_path}{number_of_qubits}_qubits/opt_{optimization_level}/'
            name = f'QPU_efficient_implementation_{number_of_qubits}_qubits_opt_{optimization_level}_{density}_density_metrics.txt'
            with open(path + name, 'r') as f:
                lines = f.readlines()
                # extract the quantum seconds and average error from the file
                for line in lines:
                    if 'quantum_seconds' in line:
                        quantum_seconds.append(float(line.split(': ')[1]))
                    if 'average_error' in line:
                        average_errors.append(float(line.split(': ')[1]))
        #plot a scatter plot of quantum seconds vs average error for the given density and optimization level
        #labels should be the density
        fig, ax = plt.subplots()
        ax.scatter(quantum_seconds, average_errors)
        for i, txt in enumerate(densities):
            ax.annotate(f"density: {txt}", (quantum_seconds[i], average_errors[i]))

        #add one red dot corresponding to quantum seconds and average error for the standard implementation
        # Load the standard implementation metrics file
        standard_name = f'QPU_standard_implementation_{number_of_qubits}_qubits_opt_{optimization_level}_metrics.txt'
        with open(path + standard_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'quantum_seconds' in line:
                    standard_quantum_seconds = float(line.split(': ')[1])
                if 'average_error' in line:
                    standard_average_error = float(line.split(': ')[1])
        
        # Add the red dot
        ax.scatter(standard_quantum_seconds, standard_average_error, color='red')
        ax.annotate(f"st_impl", (standard_quantum_seconds, standard_average_error))

        plt.xlabel('Quantum seconds')
        plt.ylabel('Average error')
        plt.title(f'QPU time vs error for different levels of density for {number_of_qubits} qubits and opt level {optimization_level}')
        
        
        #save figure to file
        plt.savefig(f'{output_path}{number_of_qubits}_qubits/opt_{optimization_level}/{number_of_qubits}_qubits_opt_{optimization_level}_scatter_plot.png')
        plt.close()