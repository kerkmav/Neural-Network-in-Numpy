import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import nnfs
from neuralnet.layers import LayerDense
from neuralnet.activations import Activation_ReLU, Activation_Softmax
from neuralnet.losses import LossCategoricalCrossEntropy

nnfs.init()

X, y = spiral_data(100, 3)

def main():
    print("Initializing Neural Network...")

    dense1 = LayerDense(2, 64, learning_rate=0.05)
    activation1 = Activation_ReLU()
    dense2 = LayerDense(64, 3, learning_rate=0.05)
    activation2 = Activation_Softmax()
    loss_function = LossCategoricalCrossEntropy()

    print("\nNetwork Summary:")
    dense1.summary("Dense1")
    dense2.summary("Dense2")
    print("\nStarting Training...\n")

    epochs = 1000
    loss_history = []
    weight_log = []

    for epoch in range(epochs):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = loss_function.calculate(activation2.output, y)
        loss_history.append(loss)

        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:04d} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
            print(f"  Dense1 weights mean: {np.mean(dense1.weights):.5f}, std: {np.std(dense1.weights):.5f}")
            print(f"  Dense2 weights mean: {np.mean(dense2.weights):.5f}, std: {np.std(dense2.weights):.5f}")

        # Backward pass
        d_values = activation2.output
        d_values[range(len(d_values)), y] -= 1
        d_values /= len(d_values)

        dense2.backward(d_values)
        activation1.backward(dense2.d_inputs)
        dense1.backward(activation1.d_inputs)

        dense1.update_params()
        dense2.update_params()

    print("\nTraining Complete ✅\nFinal Weights:")
    dense1.summary("Dense1")
    dense2.summary("Dense2")

    weight_log.append({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
        "dense1_weights_mean": np.mean(dense1.weights),
        "dense1_biases_mean": np.mean(dense1.biases),
        "dense2_weights_mean": np.mean(dense2.weights),
        "dense2_biases_mean": np.mean(dense2.biases)
    })

    # Plot
    plt.plot(loss_history)
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(weight_log)
    print("\nFinal Metrics Table:")
    print(df.tail())

    df.to_csv("training_log.csv", index=False)

if __name__ == '__main__':
    main()