import pennylane as qml
from pennylane import numpy as np

# تحديد الجهاز الكمي (المحاكي في هذه الحالة)
dev = qml.device("default.qubit", wires=1)  # كيوبت واحد

@qml.qnode(dev)
def quantum_circuit(weights):
    # تطبيق بوابة دوران (الخلايا العصبية الكمومية لدينا)
    qml.RY(weights[0], wires=0)  # weights[0] هي زاوية الدوران
    return qml.expval(qml.PauliZ(0)) # قياس الكيوبت

# تعريف دالة التكلفة بسيطة (نسعى لتقليلها)
def cost(weights):
    output = quantum_circuit(weights)
    target = 1.0  # الإخراج المطلوب
    return (output - target)**2

# تهيئة أوزان عشوائية
weights = np.random.randn(1)

# تحسين الأوزان (باستخدام نزول التدرج)
optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

for it in range(100):
    result = optimizer.step(cost, weights)
    if isinstance(result, tuple):  # Check if it's a tuple (weights and cost)
        weights, c = result
    else:  # It's just the weights
        weights = result
        c = cost(weights) # Calculate the cost manually

    if (it + 1) % 10 == 0:
        print("Cost at step {:5}: {:0.7f}".format(it + 1, c))

print("الأوزان النهائية:", weights)
print("الإخراج النهائي:", quantum_circuit(weights))