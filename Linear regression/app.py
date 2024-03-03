import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Datasetni yuklash
data_set = pd.read_csv("./Salary_Data.csv")
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 1].values

# Datasetni train va test qiymatlarga ajratib olsih 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model yaratib olish
model = LinearRegression()
model.fit(x_train, y_train)

# Test qiymatni bashorat qilish
y_pred = model.predict(x_test)

# Nuqtalarni chizish
plt.scatter(x_train, y_train, color='blue', label='Training Data')

# Nuqtalarni chizish
plt.scatter(x_test, y_test, color='red', label='Test Data')

# Regressiya chizig'i
plt.plot(x_test, y_pred, color='green', linewidth=3, label='Regression Line')

# Natija
plt.title('Linear Regression Model')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# O'rtacha kvadratik xatolik
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")