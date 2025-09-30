import qiskit_nature 
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE 
from qiskit_algorithms.optimizers import SPSA, SLSQP, COBYLA 
from qiskit_nature.second_q.transformers import FreezeCoreTransformer 
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo 
from qiskit_nature.second_q.mappers import ParityMapper 
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock 
import numpy as np 
qiskit_nature.settings.use_pauli_sum_op = False 
# pylint: disable=undefined-variable # pylint: enable=line-too-long 
from qiskit_nature.second_q.drivers import PySCFDriver 
import matplotlib.pyplot as plt 
from qiskit.circuit.library import EfficientSU2 
import time 
import tkinter as tk 
from tkinter import ttk

# Import the molecules dictionary
from molecules import molecules

#qiskit_nature.settings.use_pauli_sum_op = False

# === Functions ===
def get_qubit_op(mol_entry):
    #molecule = MoleculeInfo(
    #    symbols=mol_entry["symbols"],
    #    coords=mol_entry["coords"],
    #    multiplicity=1,
    #    charge=0,
    #)
    molecule = MoleculeInfo(
        symbols=mol_entry["symbols"],
        coords=mol_entry["coords"],
        multiplicity=1,
        charge=0,
    )
    driver = qiskit_nature.second_q.drivers.PySCFDriver.from_molecule(molecule)
    properties = driver.run()
    problem = FreezeCoreTransformer(
        freeze_core=True, remove_orbitals=[-3, -2]
    ).transform(properties)

    num_particles = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals

    mapper = ParityMapper(num_particles=num_particles)
    qubit_op = mapper.map(problem.second_q_ops()[0])
    return qubit_op, num_particles, num_spatial_orbitals, problem, mapper

def exact_solver(qubit_op, problem):
    sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
    result = problem.interpret(sol)
    return result

# === Main VQE Loop ===
optimizer = COBYLA(maxiter=100)
noiseless_estimator = Estimator(approximation=True)

start_time = time.time()

for name, mol in molecules.items():
    print(f"=== Running VQE for {name} ===")

    qubit_op, num_particles, num_spatial_orbitals, problem, mapper = get_qubit_op(mol)

    # Exact solver
    result = exact_solver(qubit_op, problem)
    exact_energy = result.total_energies[0].real

    # VQE
    init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    var_form = UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=init_state)

    vqe = VQE(
        noiseless_estimator,
        var_form,
        optimizer,
        initial_point=[0] * var_form.num_parameters,
    )
    vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
    vqe_energy = problem.interpret(vqe_calc).total_energies[0].real

    # === Log results per molecule ===
    elapsed_time = time.time() - start_time
    log_filename = f"{name}_energy_log.txt"
    with open(log_filename, "w") as log_file:
        log_file.write("Molecule\tExact Energy (Hartree)\tVQE Energy (Hartree)\n")
        log_file.write(f"{name}\t{exact_energy:.10f}\t{vqe_energy:.10f}\n")
        log_file.write(f"\nRuntime: {elapsed_time:.2f} seconds\n")

    print(f"Results logged in {log_filename}\n")

print("=== All molecules completed ===")

