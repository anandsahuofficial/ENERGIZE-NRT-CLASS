import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# --------------------------
# Generate quadratic data
# --------------------------
np.random.seed(42)
n = 50
X = np.random.uniform(-5, 5, n).reshape(-1, 1)  # random x values
y_true = 2 * X[:, 0]**2 + 3 * X[:, 0] + 5      # true quadratic function
noise = np.random.normal(0, 5, n)              # random noise
y = y_true + noise

# --------------------------
# Fit models
# --------------------------

# 1. Linear Regression


# 2. Polynomial Regression (degree 2)


# 3. Ridge Regression (with polynomial features)


# --------------------------
# Predictions for smooth curve
# --------------------------
X_plot = np.linspace(-5, 5, 200).reshape(-1, 1)


# --------------------------
# Plot results
# --------------------------
plt.scatter(X, y, color="black", label="Data (with noise)")
plt.plot(X_plot, y_lin, label="Linear Regression", color="red")
plt.plot(X_plot, y_poly, label="Polynomial Regression (deg=2)", color="blue")
plt.plot(X_plot, y_ridge, label="Ridge Regression (deg=2)", color="green")
plt.plot(X_plot, 2*X_plot[:,0]**2 + 3*X_plot[:,0] + 5,
         label="True Function", color="orange", linestyle="--")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear vs Polynomial vs Ridge Regression")
plt.show()

