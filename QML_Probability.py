from qiskit import QuantumCircuit, transpile, Aer, execute
import numpy as np

q = QuantumCircuit(1)

q.h(0)

backend = Aer.get_backend('statevector_simulator')
job = execute(q, backend)
result = job.result()
statevector = result.get_statevector(q)

print("Statevector:", statevector)

prob_0 = np.abs(statevector[0]) ** 2
prob_1 = np.abs(statevector[1]) ** 2

print("Probability of measuring |0>: ", prob_0)
print("Probability of measuring |1>: ", prob_1)
