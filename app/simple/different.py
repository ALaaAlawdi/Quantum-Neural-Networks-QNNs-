import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt  # For plotting

# 1. Classical Neural Network
def classical_neuron(x, w, b):
    return np.tanh(w * x + b)  # Using tanh activation

# 2. Quantum Neural Network
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def quantum_circuit(x):
    qml.RY(np.arcsin(x), wires=0)  # Input encoding (angle encoding)
    return qml.expval(qml.PauliZ(0))

# Example Input Data
x_data = np.linspace(-1, 1, 20)  # Create a range of input values

# 3. Training and Comparison
classical_outputs = []
quantum_outputs = []

# Classical Training (simple weight and bias adjustment)
classical_w = np.random.randn()
classical_b = np.random.randn()
learning_rate = 0.1

for x in x_data:
    classical_output = classical_neuron(x, classical_w, classical_b)
    classical_outputs.append(classical_output)

    # Simple Gradient Descent Update (for demonstration)
    error = (classical_output - np.sin(x))  # Example target function
    classical_w -= learning_rate * error * x
    classical_b -= learning_rate * error

# Quantum "Training" (adjusting the input encoding angle)
for x in x_data:
    quantum_output = quantum_circuit(x)
    quantum_outputs.append(quantum_output)


# 4. Plotting the Results
plt.figure(figsize=(10, 6))
plt.plot(x_data, np.sin(x_data), label="Target Function (sin(x))", linestyle="--")  # Example target

plt.plot(x_data, classical_outputs, label="Classical Neuron Output")
plt.plot(x_data, quantum_outputs, label="Quantum Neuron Output")

plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.title("Classical vs. Quantum Neuron")
plt.legend()
plt.grid(True)
plt.show()

print("Classical Weight:", classical_w)
print("Classical Bias:", classical_b)