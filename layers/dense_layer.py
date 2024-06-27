import numpy as np

from .layer import Layer


class DenseLayer(Layer):
    __weights: np.ndarray
    __bias: np.ndarray
    __input_data: np.ndarray

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        self.__weights = np.random.randn(input_size, output_size) * 0.05
        self.__bias = np.zeros(output_size)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.__input_data = input_data
        return np.dot(input_data, self.__weights) + self.__bias

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        reshaped_gradient: np.ndarray = output_gradient.reshape((1, output_gradient.shape[0]))
        reshaped_input_data: np.ndarray = self.__input_data.reshape(1, self.__input_data.shape[0])

        de_dx: np.ndarray = np.dot(reshaped_gradient, self.__weights.T)
        de_db: np.ndarray = output_gradient
        de_dw: np.ndarray = np.dot(reshaped_input_data.T, reshaped_gradient).reshape(self.__weights.shape)

        self.__weights -= learning_rate * de_dw
        self.__bias -= learning_rate * de_db

        return de_dx[0]
