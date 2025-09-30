import pennylane as qml
import jax
import numpy as np
import time

from pennylane import qchem
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

# Define symbols and geometry
symbols = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]
geometry = np.array([
    [ 0.695, -1.204, 0.000],
    [ 1.391,  0.000, 0.000],
    [ 0.695,  1.204, 0.000],
    [-0.695,  1.204, 0.000],
    [-1.391,  0.000, 0.000],
    [-0.695, -1.204, 0.000],
    [ 1.242, -2.152, 0.000],
    [ 2.485,  0.000, 0.000],
    [ 1.242,  2.152, 0.000],
    [-1.242,  2.152, 0.000],
    [-2.485,  0.000, 0.000],
    [-1.242, -2.152, 0.000]
])

# Re-define H using Jax Arrays
#molecule = qchem.Molecule(symbols, jnp.array(geometry))
H, qubits = qchem.molecular_hamiltonian(
    symbols,
    geometry,
    active_electrons=2,
    active_orbitals=5
)

def circuit_2(params, excitations):
    qml.BasisState(jnp.array(hf_state), wires=range(qubits))

    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(params[i], wires=excitation)
        else:
            qml.SingleExcitation(params[i], wires=excitation)
    return qml.expval(H)

#dev = qml.device("default.qubit", wires=qubit
dev = qml.device('lightning.gpu', wires=qubits) 
cost_fn = qml.QNode(circuit_2, dev, interface="jax")

active_electrons = 2
circuit_gradient = jax.grad(cost_fn, argnums=0)

singles, doubles = qchem.excitations(active_electrons, qubits)
hf_state = qchem.hf_state(active_electrons, qubits)

params = [0.0] * len(doubles)
grads = circuit_gradient(params, excitations=doubles)

for i in range(len(doubles)):
    print(f"Excitation : {doubles[i]}, Gradient: {grads[i]}")

doubles_select = [doubles[i] for i in range(len(doubles)) if abs(grads[i]) > 1.0e-5]
doubles_select

import optax

opt = optax.sgd(learning_rate=0.5)
cost_fn = qml.QNode(circuit_1, dev, interface="jax")

params = jnp.zeros(len(doubles_select))

gates_select = doubles_select
opt_state = opt.init(params)

for n in range(10):
    t1 = time.time()
    gradient = jax.grad(cost_fn, argnums=0)(params, excitations=doubles_select)
    updates, opt_state = opt.update(gradient, opt_state)
    params = optax.apply_updates(params, updates)
    energy = cost_fn(params, doubles_select)
    t2 = time.time()
    print("n = {:},  E = {:.8f} H, t = {:.2f} s".format(n, energy, t2 - t1))
