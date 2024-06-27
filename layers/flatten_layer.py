import numpy as np

from .layer import Layer


class FlattenLayer(Layer):
    __old_shape: tuple[int, ...]

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.__old_shape = input_data.shape
        return input_data.flatten()

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return output_gradient.reshape(self.__old_shape)
