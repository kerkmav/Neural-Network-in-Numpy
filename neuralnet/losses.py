import numpy as np

class Loss:
    def forward(self, output, y):
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        return np.mean(sample_loss)

class LossCategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-10, 1 - 1e-10)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clip[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clip * y_true, axis=1)

        return -np.log(correct_confidences)