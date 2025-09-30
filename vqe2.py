"""
vqe_runner.py
- Loads molecules from molecules.txt (a Python-dict style text file).
- For each molecule, builds a MoleculeInfo, runs exact solver and VQE,
  writes per-molecule energy file "<MOL>_energy_log.txt" and calls change_log.append_change().
- Minimal changes to your original code; uses Qiskit-Nature and Estimator.
"""

import ast
import time
from datetime import datetime

import qiskit_nature
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_aer.primitives import Estimator
import numpy as np

# Keep your setting
qiskit_nature.settings.use_pauli_sum_op = False

# import our change-log helper (same folder)
from change_log import append_change

# -------------------------
# Helper: load molecules.txt
# -------------------------
def load_molecules_from_txt(filename="molecules.txt"):
    """
    molecules.txt must contain a dict literal, e.g.:
    {
      "H2": {"symbols": ["H","H"], "coords": [[0,0,0],[0,0,0.74]], "multiplicity":1, "charge":0},
      "H2O": {...}
    }
    """
    with open(filename, "r") as f:
        raw = f.read()
    try:
        mol_dict = ast.literal_eval(raw)
    except Exception as e:
        raise RuntimeError(f"Failed to parse {filename}: {e}")
    return mol_dict

# -------------------------
# Build qubit operator from a molecule entry
# -------------------------
def get_qubit_op_from_entry(mol_entry):
    """
    mol_entry is dict with keys: symbols, coords, optionally multiplicity & charge.
    Returns: qubit_op, num_particles, num_spatial_orbitals, problem, mapper
    """
    multiplicity = mol_entry.get("multiplicity", 1)
    charge = mol_entry.get("charge", 0)

    molecule = MoleculeInfo(
        symbols=mol_entry["symbols"],
        coords=mol_entry["coords"],
        multiplicity=multiplicity,
        charge=charge,
    )

    driver = PySCFDriver.from_molecule(molecule)
    properties = driver.run()

    problem = FreezeCoreTransformer(
        freeze_core=True, remove_orbitals=[-3, -2]
    ).transform(properties)

    num_particles = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals
    mapper = ParityMapper(num_particles=num_particles)
    qubit_op = mapper.map(problem.second_q_ops()[0])

    return qubit_op, num_particles, num_spatial_orbitals, problem, mapper

# -------------------------
# Exact solver wrapper
# -------------------------
def exact_solver(qubit_op, problem):
    sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
    result = problem.interpret(sol)
    return result

# -------------------------
# Main runner
# -------------------------
def main(molecules_file="molecules.txt"):
    molecules = load_molecules_from_txt(molecules_file)

    optimizer = COBYLA(maxiter=100)
    noiseless_estimator = Estimator(approximation=True)

    overall_start = time.time()

    for name, mol in molecules.items():
        print(f"[{datetime.now().isoformat()}] === Running VQE for {name} ===")
        start_time = time.time()

        # prepare qubit operator
        qubit_op, num_particles, num_spatial_orbitals, problem, mapper = get_qubit_op_from_entry(mol)

        # exact energy
        exact_res = exact_solver(qubit_op, problem)
        exact_energy = exact_res.total_energies[0].real

        # prepare ansatz & VQE
        init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
        var_form = UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=init_state)

        vqe = VQE(
            noiseless_estimator,
            var_form,
            optimizer,
            initial_point=[0] * var_form.num_parameters,
        )

        # run VQE
        vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
        vqe_res = problem.interpret(vqe_calc)
        vqe_energy = vqe_res.total_energies[0].real

        elapsed = time.time() - start_time

        # write per-molecule energy log
        log_filename = f"{name}_energy_log.txt"
        with open(log_filename, "w") as f:
            f.write("Molecule\tExact Energy (Hartree)\tVQE Energy (Hartree)\n")
            f.write(f"{name}\t{exact_energy:.10f}\t{vqe_energy:.10f}\n")
            f.write(f"\nRuntime (this molecule): {elapsed:.2f} seconds\n")

        print(f"[{datetime.now().isoformat()}] Results for {name}: exact={exact_energy:.10f}, vqe={vqe_energy:.10f}")
        print(f"Logged to {log_filename} (runtime {elapsed:.2f}s)\n")

        # write entry into central change-log (append)
        append_change(
            molecule_name=name,
            exact_energy=exact_energy,
            vqe_energy=vqe_energy,
            runtime_seconds=elapsed,
            note=f"Run at {datetime.now().isoformat()}"
        )

    overall_elapsed = time.time() - overall_start
    print(f"All molecules completed. Total runtime: {overall_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

