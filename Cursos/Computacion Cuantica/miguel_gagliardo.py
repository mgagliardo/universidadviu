import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram

prob = np.sqrt([0.24, 0.56, 0.06, 0.14])

qc = QuantumCircuit(2)

qc.reset([0,1])

initial_state = prob/np.linalg.norm(prob)
qc.initialize(initial_state, [0,1])  

qc.measure_all()
display(transpile(qc, basis_gates=['ry','cx']).draw('mpl'))

backend = Aer.get_backend('aer_simulator') 

job_sim = backend.run(transpile(qc,basis_gates=['ry','cx']), shots=1024) 
result = job_sim.result()     
plot_histogram(result.get_counts())

# Para obtener los parametros de RY

from qiskit.converters import circuit_to_dag
dag = circuit_to_dag(transpile(qc, basis_gates=['ry','cx']))
gates_param = []
for node in dag.op_nodes():
    gates_param.append(node.op.params)
for i in range(qc.num_qubits):
    print(gates_param[i][0])
