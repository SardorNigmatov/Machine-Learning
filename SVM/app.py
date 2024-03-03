import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# CSV faylini Pandas DataFrame ga o'qish
df = pd.read_csv("User_Data.csv")  # "User_Data.csv" faylini o'z manzilingiz bilan almashtiring

# Xususiyatlar (X) va maqsad o'zgaruvchisi (y) ni ajratish
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Ma'lumotlarni o'qish va test qismi uchun ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xususiyatlar miqdorini tekshirish - Xususiyatlar o'zgaruvchilarini standartlash
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear kernel'li SVM model yaratish
model = SVC(kernel='linear')

# Modelni standartlangan o'qish ma'lumotlari bilan o'qitish
model.fit(X_train_scaled, y_train)

# Standartlangan test ma'lumotlariga qarab bashorat qilish
y_pred = model.predict(X_test_scaled)

# Bashoratni chiqarish
print("Aniqlik:", accuracy_score(y_test, y_pred))

# Qabul qilingan qaror chegarasini chizish
h = .02  # chiziqlik

x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Natijani rangli grafikga joylash
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

# O'qish nuqtalarini chizish
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Yosh (standartlashtirilgan)')
plt.ylabel('Baho (standartlashtirilgan)')
plt.title("SVM Qabul Qilingan Chegara va O'qish Ma\'lumotlari")
plt.show()
