import qiskit_nature
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_aer import AerSimulator
import numpy as np
import psutil
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Initialize lists for tracking
timestamps = []
counts = []
values = []
para = []
std_count = []
cpu_usages = []
ram_usages = []
core_usages = []
cumulative_times = []  # To track the cumulative time for each iteration

previous_time = 0  # Initialize previous time for cumulative time tracking

# Store intermediate results with cumulative time
def store_intermediate_result(eval_count, parameters, mean, std):
    global previous_time
    iteration_start_time = time.time()  # Record start time of this iteration

    timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    counts.append(eval_count)
    values.append(mean)
    para.append(parameters)
    std_count.append(std)

    # Calculate the cumulative time for this iteration
    cumulative_time = previous_time + (time.time() - iteration_start_time)
    cumulative_times.append(cumulative_time)
    previous_time = cumulative_time  # Update previous time for the next iteration

    # Get system resource usage
    cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
    ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
    core_count = psutil.cpu_count(logical=True)

    cpu_usages.append(cpu_percent)
    ram_usages.append(ram_used_gb)
    core_usages.append(core_count)

    # Debugging check
    #print(f"Count: {eval_count}, Time: {cumulative_time:.2f}s, CPU: {cpu_percent}, RAM: {ram_used_gb:.2f} GB")
    with open("optimization_log6.txt", "a") as f:
                     f.write(f"Count for {eval_count}")
                     f.write(f"time taken = {cumulative_time:.2f}")
                     f.write(f" CPU: {cpu_percent}, RAM: {ram_used_gb:.2f} GB")
                     f.write(f" at {time.time()}\n")

# Setup the quantum problem
driver = PySCFDriver(
    #atom = "C 0 0 0; C 1.54 0 0; O 2.97 0 0; H -0.63 0.63 0.63; H -0.63 -0.63 -0.63; H 0 -1.09 0; H 2.17 0.63 0.63; H 2.17 -0.63 -0.63; H 3.56 0.76 0; H 3.56 -0.76 0",
    #atom = "C 0.000 0.000 0.000; C 1.540 0.000 0.000; O 2.970 0.000 0.000; \
    #        H -0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.000 -0.900; \
    #        H 2.170 0.629 0.629; H 2.170 -0.629 -0.629; H 3.930 0.000 0.000",
    
    atom = "C  1.396  0.000  0.000; \
            C  0.698  1.209  0.000; \
            C -0.698  1.209  0.000; \
            C -1.396  0.000  0.000; \
            C -0.698 -1.209  0.000; \
            C  0.698 -1.209  0.000; \
            H  2.479  0.000  0.000; \
            H  1.240  2.148  0.000; \
            H -1.240  2.148  0.000; \
            H -2.479  0.000  0.000; \
            H -1.240 -2.148  0.000; \
            H  1.240 -2.148  0.000",
    basis="sto3g",
    charge=0,
    spin=0,
)

problem = driver.run()
fc_transformer = FreezeCoreTransformer()
fc_problem = fc_transformer.transform(problem)
as_transformer = ActiveSpaceTransformer(8,8)
as_problem = as_transformer.transform(fc_problem)
hamiltonian = as_problem.hamiltonian
second_q_op = hamiltonian.second_q_op()

mapper = JordanWignerMapper()
qubit_jw_op = mapper.map(second_q_op)

var_form = UCCSD(
    as_problem.num_spatial_orbitals,
    as_problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        as_problem.num_spatial_orbitals,
        as_problem.num_particles,
        mapper,
    ),
)

noiseless_estimator = Estimator()
#backend = AerSimulator(method='statevector', device='GPU')
#noiseless_estimator = backend
#optimizer = SLSQP(maxiter=100)
optimizer = COBYLA(maxiter=10)

vqe = VQE(
    noiseless_estimator,
    var_form,
    optimizer,
    initial_point=[0] * var_form.num_parameters,
    callback=store_intermediate_result
)

vqe_calc = vqe.compute_minimum_eigenvalue(qubit_jw_op)
vqe_result = problem.interpret(vqe_calc).total_energies[0].real

# Ensure that all lists have the same length before writing to the CSV
min_length = min(len(timestamps), len(counts), len(values), len(ram_usages), len(core_usages))

# Save data to CSV file
output_file = "output.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['Evaluation Count', 'Timestamp', 'Cumulative Time (s)', 'Energy', 'RAM Usage (GB)', 'Core Count']
    num_cores = len(cpu_usages[0]) if cpu_usages else 0
    header += [f'CPU Usage Core {i}' for i in range(num_cores)]
    writer.writerow(header)

    for i in range(min_length):
        row = [counts[i], timestamps[i], cumulative_times[i], values[i], ram_usages[i], core_usages[i]]
        row.extend(cpu_usages[i])  # Append CPU usage for each core
        writer.writerow(row)

print(f"System resource data saved to {output_file}")

# Annotating and plotting
def annotate_min_max_energy(counts, values):
    min_energy_index = values.index(min(values))
    max_energy_index = values.index(max(values))

    plt.plot(counts[min_energy_index], values[min_energy_index], 'o', markerfacecolor='green', markeredgewidth=2, markersize=5, label='Min Energy')
    plt.plot(counts[max_energy_index], values[max_energy_index], 'o', markerfacecolor='red', markeredgewidth=2, markersize=5, label='Max Energy')

    plt.annotate(f"Min: {min(values):.6f}", (counts[min_energy_index], values[min_energy_index]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.annotate(f"Max: {max(values):.6f}", (counts[max_energy_index], values[max_energy_index]), textcoords="offset points", xytext=(0, 10), ha='center')

# Plot exact energies
plt.figure(figsize=(10, 6))
plt.plot(counts, values, label='Exact Energies')
annotate_min_max_energy(counts, values)
plt.xlabel('Evaluation Count')
plt.ylabel('Exact Energy')
plt.title('VQE Exact Energies')
plt.legend()
plt.savefig("vqe_plot.png")  # Save the VQE plot
plt.close()

print(f"VQE plot saved as vqe_plot.png")
