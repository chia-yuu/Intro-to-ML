import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        # raise NotImplementedError
        x = np.c_[np.ones([X.shape[0], 1]), X]
        # (Xt X)^-1 Xt y
        inv = np.linalg.inv(np.dot(x.T, x))
        beta = np.dot(np.dot(inv, x.T), y)
        self.intercept = beta[0]
        self.weights = beta[1:]

    def predict(self, X):
        # raise NotImplementedError
        return np.dot(X, self.weights) + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        # raise NotImplementedError
        # Wt+1 = Wt + learningRate * LossFuncDifferenciate
        # init
        self.weights = np.zeros(X.shape[1])
        # self.weights = np.random.randn(X.shape[1]) * 0.01
        self.intercept = 0
        n = len(y)
        tot_loss = []
        # run
        for ep in range(epochs):
            # if(ep%100000 == 0):print(f"ep {ep}", end=' ')
            predict_y = self.predict(X)
            loss = compute_mse(predict_y, y)
            tot_loss.append(loss)
            # if(ep%10000 == 0):print(f"loss = {loss}, predict y = {predict_y}, y = {y}")

            diff_w = (2 / n) * np.dot(X.T, (predict_y - y))
            diff_inter = (2 / n) * np.sum(predict_y - y)
            # if(ep%10000 == 0):print(f"dif_w = {diff_w}, dif_inter = {diff_inter}")

            self.weights = self.weights - learning_rate * diff_w
            self.intercept = self.intercept - learning_rate * diff_inter
        return tot_loss

    def predict(self, X):
        # raise NotImplementedError
        # if(ep%10000 == 0):
        #     print(f"weight = {self.weights}")
        #     print(f"predict dot = {np.dot(X, self.weights)}")
        #     print(f"np dot = {np.dot(X, self.weights) + self.intercept}")
        return np.dot(X, self.weights) + self.intercept

    def plot_learning_curve(self, loss):
        # raise NotImplementedError
        plt.plot(range(len(loss)), loss)
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("MES Loss")
        plt.show()


def compute_mse(prediction, ground_truth):
    # raise NotImplementedError
    n = len(prediction)
    return (1 / n) * np.sum((prediction - ground_truth)**2)


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    # losses = LR_GD.fit(train_x, train_y, learning_rate=1e-2, epochs=1000)
    # losses = LR_GD.fit(train_x, train_y, learning_rate=5e-5, epochs=10000000)
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=1000000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Mean prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
