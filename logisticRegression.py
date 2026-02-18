import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# --- Model Definition ---
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.05, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        weights_array = np.where(y == 1, 2.0, 1.0)

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            dw = (1 / m) * np.dot(X.T, (y_pred - y) * weights_array)
            db = (1 / m) * np.sum((y_pred - y) * weights_array)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 1000 == 0:
                loss = -np.mean(weights_array * (y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15)))
                print(f"Epoch {epoch}: loss = {loss:.4f}")

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)


# --- Data Preparation ---
df = pd.read_csv("heart.csv")
df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != 'object' else x)

X_raw = df.drop(columns=["TenYearCHD"]).values
y = df["TenYearCHD"].values

np.random.seed(67)
indices = np.random.permutation(len(X_raw))
train_size = int(0.8 * len(X_raw))
X_train_raw, y_train = X_raw[indices[:train_size]], y[indices[:train_size]]
X_test_raw, y_test = X_raw[indices[train_size:]], y[indices[train_size:]]

mean, std = np.mean(X_train_raw, axis=0), np.std(X_train_raw, axis=0)
std[std == 0] = 1
X_train, X_test = (X_train_raw - mean) / std, (X_test_raw - mean) / std

# --- Execution ---
model = LogisticRegressionScratch(learning_rate=0.05, epochs=5000)
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)
y_pred = (y_probs >= 0.5).astype(int)

print(f"\nAccuracy: {np.mean(y_pred == y_test):.4f}")
print(classification_report(y_test, y_pred))

# --- Visualization ---
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Balanced Confusion Matrix")
plt.show()