import numpy as np

from layers import Layer


class Model:
    __layers: list[Layer]

    def __init__(self, layers: list[Layer]):
        self.__layers = layers

    def predict(self, image: np.ndarray, threshold: float) -> [int]:
        result: list[int] = []
        forward_value: np.ndarray = self.__forward(image)

        for i in range(len(forward_value)):
            if forward_value[i] > threshold:
                result.append(i)

        return result

    def train(self, image: np.ndarray, true_label: int, learning_rate: float) -> float:
        """
        Returns cross-entropy error.
        """

        y_pred: np.ndarray = self.__forward(image)
        y_true: np.ndarray = self.__get_y_true(true_label)

        cross_entropy: float = self.__cross_entropy(y_true=y_true, y_pred=y_pred)
        gradient: np.ndarray = y_pred - y_true  # This magic works because model uses SoftMax and Cross-Entropy
        self.__backward(gradient=gradient, learning_rate=learning_rate)

        return cross_entropy

    def __forward(self, image: np.ndarray) -> np.ndarray:
        result: np.ndarray = image

        for i, layer in enumerate(self.__layers):
            result = layer.forward(result)

        return self.__softmax(result)

    def __backward(self, gradient: np.ndarray, learning_rate: float) -> None:
        current_gradient: np.ndarray = gradient
        for layer in self.__layers[::-1]:
            current_gradient = layer.backward(current_gradient, learning_rate)

    @staticmethod
    def __get_y_true(label: int) -> np.ndarray:
        result: np.ndarray = np.zeros(10)
        result[label] = 1
        return result

    @staticmethod
    def __softmax(data: np.ndarray) -> np.ndarray:
        exps: np.ndarray = np.exp(data)
        return exps / exps.sum()

    @staticmethod
    def __cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        result: float = 0

        for i in range(len(y_true)):
            result -= y_true[i] * np.log2(y_pred[i])

        return result

    @staticmethod
    def __cross_entropy_gradient(y_pred_softmax: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return y_pred_softmax - y_true
