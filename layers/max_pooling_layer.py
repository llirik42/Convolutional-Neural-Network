import numpy as np
from numba import jit

from .layer import Layer


class MaxPoolingLayer(Layer):
    __size: int
    __input_shape: tuple[int, ...]
    __output_shape: tuple[int, ...]
    __indexes_of_max: np.ndarray

    def __init__(self, input_shape: tuple[int, ...], size: int):
        self.__size = size
        self.__input_shape = input_shape
        self.__output_shape = tuple([input_shape[0]] + [input_shape[i] // size for i in range(1, len(input_shape))])
        self.__indexes_of_max = np.empty(self.__output_shape, dtype=int)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.empty(self.__output_shape)

        for i in range(self.__output_shape[0]):
            self.__indexes_of_max[i], result[i] = self.__max_pooling(
                input_data=input_data[i], pooling_size=self.__size, output_shape=self.__output_shape
            )

        return result

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return self.__calculate_gradient(
            output_gradient=output_gradient,
            pooling_size=self.__size,
            input_shape=self.__input_shape,
            output_shape=self.__output_shape,
            indexes_of_max=self.__indexes_of_max,
        )

    @staticmethod
    @jit(nopython=True)
    def __max_pooling(
        input_data: np.ndarray, pooling_size: int, output_shape: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        old_shape: tuple[int, ...] = input_data.shape
        new_shape: tuple[int, ...] = (old_shape[0] // pooling_size, old_shape[1] // pooling_size)
        result: np.ndarray = np.empty(new_shape)
        indexes_of_max: np.ndarray = np.empty(output_shape[1:])

        for i in range(old_shape[0] // pooling_size):
            i1: int = i * pooling_size
            i2: int = (i + 1) * pooling_size

            for j in range(old_shape[1] // pooling_size):
                block: np.ndarray = input_data[i1:i2, j * pooling_size : (j + 1) * pooling_size]
                indexes_of_max[i, j] = block.argmax()
                result[i, j] = np.max(block)

        return indexes_of_max, result

    @staticmethod
    @jit(nopython=True)
    def __calculate_gradient(
        output_gradient: np.ndarray,
        pooling_size: int,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        indexes_of_max: np.ndarray,
    ) -> np.ndarray:
        result: np.ndarray = np.empty(input_shape)

        for k in range(output_shape[0]):
            for i in range(output_shape[1]):
                for j in range(output_shape[2]):
                    index_of_max: int = indexes_of_max[k, i, j]
                    i1: int = index_of_max % pooling_size
                    i2: int = index_of_max // pooling_size
                    result[k, i * pooling_size + i2, j * pooling_size + i1] = output_gradient[k, i, j]

        return result
