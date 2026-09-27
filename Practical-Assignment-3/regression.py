
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


data = {
    "Area": [
        800, 900, 1000, 1100, 1200,
        1300, 1400, 1500, 1600, 1700,
        1800, 1900, 2000, 2100, 2200,
        2300, 2400, 2500, 2600, 2700
    ],
    "Price": [
        25, 28, 30, 32, 35,
        37, 40, 42, 45, 47,
        50, 52, 55, 57, 60,
        62, 65, 67, 70, 72
    ]
}

df = pd.DataFrame(data)

X = df[["Area"]].values
y = df["Price"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model 1: Scikit-learn Regression
# -------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------
# Model 2: Gradient Descent
# -------------------------------


mean_x = X_train.mean()
std_x = X_train.std()

X_train_scaled = (X_train.flatten() - mean_x) / std_x
X_test_scaled = (X_test.flatten() - mean_x) / std_x


m = 0.0
c = 0.0

learning_rate = 0.01
iterations = 1000
n = len(X_train_scaled)

cost_history = []

for i in range(iterations):
    y_gd = m * X_train_scaled + c

    error = y_gd - y_train

    dm = (2 / n) * np.sum(error * X_train_scaled)
    dc = (2 / n) * np.sum(error)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

    cost = np.mean((y_train - (m * X_train_scaled + c)) ** 2)
    cost_history.append(cost)


y_gd_pred = m * X_test_scaled + c


gd_slope = m / std_x
gd_intercept = c - (m * mean_x / std_x)

print("SCIKIT-LEARN LINEAR REGRESSION")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nGRADIENT DESCENT")
print("Slope:", gd_slope)
print("Intercept:", gd_intercept)
print("Final Training MSE:", cost_history[-1])

# -------------------------------
# Performance Evaluation
# -------------------------------

def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    print("\n", name)
    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    return mae, mse, rmse, r2

evaluate_model("Scikit-learn Model", y_test, y_pred)
evaluate_model("Gradient Descent Model", y_test, y_gd_pred)


area = 1500

price_sklearn = model.predict([[area]])[0]
price_gd = gd_slope * area + gd_intercept

print("\nHOUSE PRICE PREDICTION")
print("Area:", area, "sq. ft.")
print("Scikit-learn Prediction:", round(price_sklearn, 2), "lakhs")
print("Gradient Descent Prediction:", round(price_gd, 2), "lakhs")

# -------------------------------
# Graph 1: Actual vs Predicted
# -------------------------------

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, label="Predicted Prices")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    label="Perfect Prediction"
)
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Graph 2: Gradient Descent Cost
# -------------------------------

plt.figure(figsize=(7, 5))
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Cost Reduction")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Graph 3: Regression Line
# -------------------------------

plt.figure(figsize=(7, 5))
plt.scatter(X, y, label="Actual Data")

x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.plot(x_line, y_line, label="Regression Line")
plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price (Lakhs)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()