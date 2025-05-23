import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression():

    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
            
    def fit(self, X, y):
        self.m, self.n = X.shape # m: number of data, n: number of features

        # weight:
        self.w = np.zeros(self.n)

        self.b = 0.0

        self.X = X

        self.y = y

        # Gradient descent
        for epoch in range(self.epochs):
            if epoch % 100 == 0:
                y_pred = self.predict(self.X)
                loss = self.compute_loss(y_pred, self.y)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
            self.update_weights()
        
    def update_weights(self):
        y_pred = self.predict(self.X)

        dw = (1 / self.m) * np.dot(self.X.T, (y_pred - self.y))
        db = (1 / self.m) * np.sum(y_pred - self.y)

        self.w -= self.learning_rate* dw
        self.b -= self.learning_rate* db
    
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def predict(self, X):
        return np.dot(X, self.w) + self.b
        
def main():

    df = pd.read_csv("data/Salary_dataset.csv")

    X = df[["YearsExperience"]].values
    y = df["Salary"].values

    model = LinearRegression(0.01, 1000)
    model.fit(X, y)

    pred = model.predict(X)

    plt.scatter(X, y, label="Actual")
    plt.plot(X, pred, color="red", label="Prediction")
    plt.legend()
    plt.xlabel("YearsExperience")
    plt.ylabel("Salary")
    plt.title("Linear Regression Fit")
    plt.show()


if __name__ == "__main__" : 
    
    main()

