import numpy as np

from .layer import Layer


class PaddingLayer(Layer):
    __padding_width: int

    def __init__(self, padding_width: int):
        self.__padding_width = padding_width

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        input_shape: tuple[int, ...] = input_data.shape
        new_shape: tuple[int, ...] = self.__get_new_shape(old_shape=input_shape, delta=self.__padding_width * 2)
        result = np.empty(new_shape)

        for i in range(input_shape[0]):
            result[i] = np.pad(array=input_data[i], pad_width=self.__padding_width)

        return result

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        old_shape: tuple[int, ...] = output_gradient.shape
        new_shape: tuple[int, ...] = PaddingLayer.__get_new_shape(old_shape=old_shape, delta=-self.__padding_width * 2)
        result: np.ndarray = np.empty(new_shape)

        for i in range(new_shape[0]):
            result[i] = PaddingLayer.__unpad(x=output_gradient[i], padding_width=self.__padding_width)

        return result

    @staticmethod
    def __get_new_shape(old_shape: tuple[int, ...], delta: int) -> tuple[int, ...]:
        new_shape: list[int] = [old_shape[0]]

        for i in range(1, len(old_shape)):
            new_shape.append(old_shape[i] + delta)

        return tuple(new_shape)

    @staticmethod
    def __unpad(x: np.ndarray, padding_width: int) -> np.ndarray:
        if len(x.shape) == 1:
            return x[padding_width:-padding_width]

        new_shape: tuple[int, ...] = tuple([d - padding_width * 2 for d in x.shape])
        result: np.ndarray = np.empty(new_shape)

        for i in range(new_shape[0]):
            result[i] = PaddingLayer.__unpad(x=x[i + padding_width], padding_width=padding_width)

        return result
