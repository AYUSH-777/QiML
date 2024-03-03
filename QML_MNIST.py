import pennylane as qml
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev)
def quantum_feature_map(x):
    for i in range(4):
        qml.RY(x[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

X_train_quantum = np.array([quantum_feature_map(x) for x in X_train])

classifier = SVC(kernel='linear')
classifier.fit(X_train_quantum, y_train)

X_test_quantum = np.array([quantum_feature_map(x) for x in X_test])

y_test_pred = classifier.predict(X_test_quantum)

test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

