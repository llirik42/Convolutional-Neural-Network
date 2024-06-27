import numpy as np
from scipy import signal

from .layer import Layer


class ConvolutionLayer(Layer):
    __input_depth: int
    __output_depth: int
    __input_shape: tuple[int, int, int]
    __kernels_shape: tuple[int, int, int, int]
    __kernels: np.ndarray
    __bias: np.ndarray
    __input_data: np.ndarray

    def __init__(self, input_shape: tuple[int, int, int], kernel_size: int, output_depth: int):
        input_depth, input_height, input_width = input_shape
        output_shape: tuple[int, int, int] = (
            output_depth,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        )

        self.__input_depth = input_depth
        self.__output_depth = output_depth
        self.__input_shape = input_shape
        self.__kernels_shape = (output_depth, input_depth, kernel_size, kernel_size)
        self.__kernels = np.random.randn(*self.__kernels_shape) * 0.05
        self.__bias = np.random.randn(*output_shape) * 0.05

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.__input_data = input_data
        output_data: np.ndarray = np.copy(self.__bias)

        for i in range(self.__output_depth):
            for j in range(self.__input_depth):
                output_data[i] += signal.correlate2d(in1=self.__input_data[j], in2=self.__kernels[i, j], mode="valid")

        return output_data

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        kernels_gradient: np.ndarray = np.empty(self.__kernels_shape)
        input_gradient: np.ndarray = np.zeros(self.__input_shape)

        for i in range(self.__output_depth):
            for j in range(self.__input_depth):
                kernels_gradient[i, j] = signal.correlate2d(
                    in1=self.__input_data[j], in2=output_gradient[i], mode="valid"
                )
                input_gradient[j] += signal.convolve2d(in1=output_gradient[i], in2=self.__kernels[i, j], mode="full")

        self.__kernels -= learning_rate * kernels_gradient
        self.__bias -= learning_rate * output_gradient

        return input_gradient
