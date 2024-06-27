import os
import pickle as pkl

import numpy as np
from tensorflow.keras.datasets import mnist
from tqdm import tqdm

from layers import (
    ImageNormalizationLayer,
    ConvolutionLayer,
    PaddingLayer,
    AveragePoolingLayer,
    FlattenLayer,
    DenseLayer,
)
from metrics import print_metrics
from model import Model

LEARNING_RATE = 0.001
EPOCHS = 5
MODEL_PATH: str = "model.pkl"


def get_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    return mnist.load_data()


def get_trained_model(x_train: np.ndarray, y_train: np.ndarray) -> Model:
    if os.path.isfile(MODEL_PATH):
        print(f'Loading model from "{MODEL_PATH}" ...')
        with open(MODEL_PATH, "rb") as model_file:
            result: Model = pkl.load(model_file)
            print("Model is loaded!\n")
            return result

    result = Model(
        layers=[
            ImageNormalizationLayer(),
            PaddingLayer(padding_width=2),
            ConvolutionLayer(
                input_shape=(1, 32, 32),
                kernel_size=5,
                output_depth=6,
            ),
            AveragePoolingLayer(
                input_shape=(6, 28, 28),
                size=2,
            ),
            ConvolutionLayer(input_shape=(6, 14, 14), kernel_size=5, output_depth=16),
            AveragePoolingLayer(
                input_shape=(16, 10, 10),
                size=2,
            ),
            FlattenLayer(),
            DenseLayer(
                input_size=400,
                output_size=120,
            ),
            DenseLayer(input_size=120, output_size=84),
            DenseLayer(
                input_size=84,
                output_size=10,
            ),
        ]
    )

    train(
        model=result,
        x_train=x_train,
        y_train=y_train,
    )

    print(f'Dumping model to "{MODEL_PATH}" ...')
    with open(MODEL_PATH, "wb") as model_file:
        pkl.dump(result, model_file)
        print("Model is dumped\n")

    return result


def reshape_image_for_model(data: np.ndarray) -> np.ndarray:
    return data.reshape((1, 28, 28))


def train(model: Model, x_train: np.ndarray, y_train: np.ndarray) -> None:
    print("Training model ...")

    for j in range(EPOCHS):
        avg_cross_entropy: float = 0
        for i in tqdm(range(len(x_train))):
            x: np.ndarray = reshape_image_for_model(x_train[i])
            avg_cross_entropy += model.train(
                image=x,
                true_label=y_train[i],
                learning_rate=LEARNING_RATE,
            )
        print(f"EPOCH {j + 1} / {EPOCHS}. Average cross-entropy = {avg_cross_entropy / len(x_train)}")

    print("Training done!\n")


def test(model: Model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    number_of_digits: int = 10
    tp: np.ndarray = np.zeros(number_of_digits)
    fp: np.ndarray = np.zeros(number_of_digits)
    tn: np.ndarray = np.zeros(number_of_digits)
    fn: np.ndarray = np.zeros(number_of_digits)
    support: np.ndarray = np.zeros(number_of_digits, dtype=int)

    print("Testing model ...")
    for i in tqdm(range(len(x_test))):
        x: np.ndarray = reshape_image_for_model(x_test[i])
        pred_labels: list[int] = model.predict(x, threshold=0.5)
        true_label: int = y_test[i]
        support[true_label] += 1

        for digit in range(number_of_digits):
            is_true: bool = digit == true_label
            is_predicted: bool = digit in pred_labels

            if is_true and is_predicted:
                tp[digit] += 1
            if not is_true and not is_predicted:
                tn[digit] += 1
            if not is_true and is_predicted:
                fp[digit] += 1
            if is_true and not is_predicted:
                fn[digit] += 1
    print("Testing done!\n")

    print_metrics(tp=tp, tn=tn, fp=fp, fn=fn, support=support)


def main() -> None:
    (x_train, y_train), (x_test, y_test) = get_mnist()

    model: Model = get_trained_model(x_train=x_train, y_train=y_train)

    test(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
