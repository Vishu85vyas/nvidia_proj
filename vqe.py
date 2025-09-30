import qiskit_nature
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.transformers import FreezeCoreTransformer , ActiveSpaceTransformer
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
import numpy as np
qiskit_nature.settings.use_pauli_sum_op = False
from qiskit_nature.second_q.drivers import PySCFDriver
import time
from qiskit_aer.primitives import Estimator
from pyscf import gto

# === utility: read molecule dict from file ===
import ast
def load_molecules(file_path="molecules5.txt"):
    with open(file_path, "r") as f:
        text = f.read()
    return ast.literal_eval(text)   # safely convert string → dict

molecules_dict = load_molecules()

def get_qubit_op(mol_entry, dist=None):
    """Build Qubit operator for any molecule entry"""
    symbols = mol_entry["symbols"]
    coords = mol_entry["coords"]

    # if distance param is provided, apply it to first atom
    #if dist is not None:
    #    coords = [(x, y, z) if i == 0 else (x, y, z)
    #              for i, (x, y, z) in enumerate(coords)]
        # replace placeholder "dist" in coords with actual numeric value
    if dist is not None:
        coords = [
            tuple(dist if v == "dist" else v for v in point)
            for point in coords
        ]
       # print("coordinates = " ,coords)
    molecule = MoleculeInfo(
        symbols=symbols,
        coords=coords,
        multiplicity=mol_entry.get("multiplicity", 1),
        charge=mol_entry.get("charge", 0),
    )

    driver = PySCFDriver.from_molecule(molecule)
    properties = driver.run()
    problem = FreezeCoreTransformer(
        freeze_core=True, remove_orbitals=[-3, -2]
    ).transform(properties)
    #transformer = ActiveSpaceTransformer(4,4)
    #es_problem_reduced = transformer.transform(problem)

    num_particles = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals

    mapper = ParityMapper(num_particles=num_particles)
    qubit_op = mapper.map(problem.second_q_ops()[0])

    return qubit_op, num_particles, num_spatial_orbitals, problem, mapper

def exact_solver(qubit_op, problem):
    sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
    return problem.interpret(sol)

# === VQE loop for all molecules ===
start_time = time.time()
distances = np.arange(0.5, 2.0, 0.1)
optimizer = COBYLA(maxiter=100)
noiseless_estimator = Estimator(approximation=True)

log_filename = "energy_log2.txt"
with open(log_filename, "w") as log_file:
    for mol_name, mol_entry in molecules_dict.items():
        log_file.write(f"\n==== {mol_name} ====\n")
        exact_energies, vqe_energies = [], []

        for dist in distances:
            qubit_op, num_particles, num_spatial_orbitals, problem, mapper = get_qubit_op(mol_entry, dist)

            # exact solver
            start_time = time.time()
            result = exact_solver(qubit_op, problem)
            end_time = time.time()
            tot_time = end_time - start_time   
            exact_energies.append(result.total_energies[0].real)

            # log timing to a separate file
            with open("time1.txt", "a") as tfile:
              tfile.write(
                  f"molecule name = {mol_name}, "
                  f"distance = {dist:.2f}, "
                  f"tot time in this iter = {tot_time:.6f} seconds\n"
               )

            # Your callback function
            counts = []
            values = []
            def store_intermediate_result(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)
                # --- liog to file ---
                with open("optimization_log3.txt", "a") as f:
                     f.write(f"results for {mol_name}")    
                     f.write(f"{eval_count},{mean}\n")

            # VQE
            init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
            var_form = UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=init_state)
            vqe = VQE(noiseless_estimator, var_form, optimizer, initial_point=[0] * var_form.num_parameters,callback=store_intermediate_result)
            vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
            vqe_result = problem.interpret(vqe_calc).total_energies[0].real
            vqe_energies.append(vqe_result)

        # log molecule results
        log_file.write("Distance (Å)\tExact Energy (Ha)\tVQE Energy (Ha)\n")
        for d, e_exact, e_vqe in zip(distances, exact_energies, vqe_energies):
            log_file.write(f"{d:.2f}\t{e_exact:.10f}\t{e_vqe:.10f}\n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f" All molecules processed. Results in {log_filename}")
print(f" Total runtime: {elapsed_time:.2f} sec")

