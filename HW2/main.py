import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[np.float_],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        # raise NotImplementedError
        # init
        n_data, n_feature = inputs.shape
        self.weights = np.zeros(n_feature)
        self.intercept = 0

        # run
        for it in range(self.num_iterations):
            model = np.dot(inputs, self.weights) + self.intercept
            predict_y = self.sigmoid(model)

            dw = (1 / n_data) * np.dot(inputs.T, (predict_y - targets))
            di = (1 / n_data) * np.sum(predict_y - targets)

            self.weights = self.weights - self.learning_rate * dw
            self.intercept = self.intercept - self.learning_rate * di

    def predict(
        self,
        inputs: npt.NDArray[np.float_],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        # raise NotImplementedError
        model = np.dot(inputs, self.weights) + self.intercept
        prob = self.sigmoid(model)
        cla = [1 if p >= 0.5 else 0 for p in prob]
        return prob, cla

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        # raise NotImplementedError
        return 1 / (1 + np.exp(-x))


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[np.float_],
        targets: t.Sequence[int],
    ) -> None:
        # raise NotImplementedError
        c0 = inputs[targets == 0]
        c1 = inputs[targets == 1]
        self.m0 = np.mean(c0, axis=0)
        self.m1 = np.mean(c1, axis=0)
        # self.sw = np.cov(c0.T) + np.cov(c1.T)
        self.sw = np.zeros((2, 2))
        for x in c0:
            self.sw += np.dot((x - self.m0).reshape(2, 1), (x - self.m0).reshape(1, 2))
        for x in c1:
            self.sw += np.dot((x - self.m1).reshape(2, 1), (x - self.m1).reshape(1, 2))
        # print(f"m0 = {self.m0}\nm1 = {self.m1}\nsw = {self.sw}")
        self.sb = np.dot((self.m0 - self.m1).reshape(2, 1), ((self.m0 - self.m1).reshape(1, 2)))
        # print(f"sb = {self.sb}")
        # print(f"sw_inv shape = {np.linalg.inv(self.sw).shape}")
        self.w = np.linalg.inv(self.sw).dot((self.m0 - self.m1))
        self.slope = self.w[1] / self.w[0]

    def predict(
        self,
        inputs: npt.NDArray[np.float_],
    ) -> t.Sequence[t.Union[int, bool]]:
        # raise NotImplementedError
        proj = np.dot(inputs, self.w)
        threshold = (np.dot(self.m0, self.w) + np.dot(self.m1, self.w)) / 2
        return (proj < threshold).astype(int)

    def plot_projection(self, inputs: npt.NDArray[np.float_], targets: t.Sequence[int]):
        # raise NotImplementedError
        pred = self.predict(inputs)
        c0 = inputs[pred == 0]
        c1 = inputs[pred == 1]
        # cc0 = inputs[targets == 0]
        # cc1 = inputs[targets == 1]
        x = np.linspace(-1.2, 1.2, 10)
        y = self.slope * x
        plt.plot(c0[:, 0], c0[:, 1], '.', c='red')
        plt.plot(c1[:, 0], c1[:, 1], '.', c='green')
        plt.plot(x, y, c='blue')
        for p in c0:
            a = (p[1] + (p[0] / self.slope)) / (self.slope + 1 / self.slope)
            b = self.slope * a
            plt.plot(a, b, '.', c='red', alpha=0.3)
            plt.plot([p[0], a], [p[1], b], c='red', alpha=0.1)
            # if(flag): print(a, b); flag = False
        for p in c1:
            a = (p[1] + (p[0] / self.slope)) / (self.slope + 1 / self.slope)
            b = self.slope * a
            plt.plot(a, b, '.', c='green', alpha=0.3)
            plt.plot([p[0], a], [p[1], b], c='green', alpha=0.1)
            # if(not flag): print(a, b); flag = True
        plt.text((plt.xlim()[0] + plt.xlim()[1]) / 2, plt.ylim()[1],
                 f'Projection Line: w = {self.slope}, b = {0}', va='bottom', ha='center')
        plt.show()


def compute_auc(y_trues, y_preds):
    # raise NotImplementedError
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds):
    # raise NotImplementedError
    correct = np.sum(y_trues == y_preds)
    tot = len(y_trues)
    if (tot != 0):
        return correct / tot
    else:
        return 0.0


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-4,  # You can modify the parameters as you want
        num_iterations=1000000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    print(f"x train: {x_train.shape}")
    print(f"x test: {x_test.shape}")

    FLD_.fit(x_train, y_train)
    y_pred_classes = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_train, y_train)


if __name__ == '__main__':
    main()
