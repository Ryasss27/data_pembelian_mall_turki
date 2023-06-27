import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset (ganti dengan dataset yang berbeda)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Buat objek LinearRegression
regressor = LinearRegression()

# Melakukan fitting pada data
regressor.fit(X, y)

# Mencetak koefisien regresi (slope)
print("Koefisien: ", regressor.coef_)

# Mencetak intercept regresi
print("Intercept: ", regressor.intercept_)

# Prediksi nilai untuk data baru
new_X = np.array([[6], [7]])
predicted_y = regressor.predict(new_X)
print("Prediksi: ", predicted_y)
