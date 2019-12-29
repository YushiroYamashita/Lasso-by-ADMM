import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin  #回帰に使われるクラス

class Admm(BaseEstimator, RegressorMixin):  #ADMMを用いたLassoのクラス
    def __init__(self, lambd=1.0, rho=1.0, max_iter=1000):
        self.lambd = lambd
        self.rho = rho
        self.threshold = lambd / rho
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = 0.0

    def _soft_threshold(self, x):   #軟閾値関数
        t = self.threshold

        positive_indexes = x >= t
        negative_indexes = x <= t
        zero_indexes = abs(x) <= t

        y = np.zeros(x.shape)
        y[positive_indexes] = x[positive_indexes] - t
        y[negative_indexes] = x[negative_indexes] + t
        y[zero_indexes] = 0.0

        return y

    def fit(self, X, y):    #係数の更新
        N = X.shape[0]
        M = X.shape[1]
        inv_matrix = np.linalg.inv(np.dot(X.T, X) / N + self.rho * np.identity(M))

        beta = np.dot(X.T, y) / N
        theta = beta.copy()
        mu = np.zeros(len(beta))

        for iteration in range(self.max_iter):
            beta = np.dot(inv_matrix, np.dot(X.T, y) / N + self.rho * theta - mu)
            theta = self._soft_threshold(beta + mu / self.rho)
            mu += self.rho * (beta - theta)

        self.coef_ = beta

        return self

    def predict(self, X):   #予測値の出力
        y = np.dot(X, self.coef_)
        return y
