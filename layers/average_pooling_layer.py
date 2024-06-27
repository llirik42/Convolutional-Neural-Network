import numpy as np
from numba import jit

from .layer import Layer


class AveragePoolingLayer(Layer):
    __size: int
    __input_shape: tuple[int, ...]
    __output_shape: tuple[int, ...]

    def __init__(self, input_shape: tuple[int, ...], size: int):
        self.__size = size
        self.__input_shape = input_shape
        self.__output_shape = tuple([input_shape[0]] + [input_shape[i] // size for i in range(1, len(input_shape))])

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.empty(self.__output_shape)

        for i in range(self.__output_shape[0]):
            result[i] = self.__avg_pooling(input_data=input_data[i], pooling_size=self.__size)

        return result

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return self.__calculate_gradient(
            output_gradient=output_gradient,
            pooling_size=self.__size,
            input_shape=self.__input_shape,
        )

    @staticmethod
    @jit(nopython=True)
    def __avg_pooling(input_data: np.ndarray, pooling_size: int) -> np.ndarray:
        old_shape: tuple[int, ...] = input_data.shape
        new_shape: tuple[int, ...] = (old_shape[0] // pooling_size, old_shape[1] // pooling_size)
        result: np.ndarray = np.empty(new_shape)

        block_size: int = pooling_size**2
        width: int = old_shape[1] // pooling_size
        for i in range(old_shape[0] // pooling_size):
            i1: int = i * pooling_size
            i2: int = (i + 1) * pooling_size

            for j in range(width):
                block: np.ndarray = input_data[i1:i2, j * pooling_size : (j + 1) * pooling_size]
                result[i, j] = np.sum(block) / block_size

        return result

    @staticmethod
    @jit(nopython=True)
    def __calculate_gradient(
        output_gradient: np.ndarray, pooling_size: int, input_shape: tuple[int, ...]
    ) -> np.ndarray:
        block_size: int = pooling_size**2

        result: np.ndarray = np.empty(input_shape)
        adjusted_gradient: np.ndarray = output_gradient / block_size

        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                i1: int = j // pooling_size

                for k in range(input_shape[2]):
                    i2: int = k // pooling_size
                    result[i, j, k] = adjusted_gradient[i, i1, i2]

        return result
