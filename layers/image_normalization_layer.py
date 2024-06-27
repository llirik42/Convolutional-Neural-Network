import numpy as np

from .layer import Layer


class ImageNormalizationLayer(Layer):
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return input_data / 255

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        pass  # It's used as very first layer, so we don't have to return gradient
