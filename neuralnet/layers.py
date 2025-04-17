import numpy as np

class LayerDense:
    def __init__(self, n_inputs, n_neurons, learning_rate=1.0):
        # Weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.learning_rate = learning_rate

        # Initialize for clarity and type inference
        self.inputs = None
        self.output = None
        self.d_weights = None
        self.d_biases = None
        self.d_inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        # Gradients
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        self.d_inputs = np.dot(d_values, self.weights.T)

    def update_params(self):
        self.weights -= self.learning_rate * self.d_weights
        self.biases -= self.learning_rate * self.d_biases

    def summary(self, name="Layer"):
        print(f"{name} | Weights shape: {self.weights.shape}, Biases shape: {self.biases.shape}")