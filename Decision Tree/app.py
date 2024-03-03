import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ma'lumotlarni olish
df = pd.read_csv("./User_Data.csv")

# Xususiyatlarni va maqsad o'zgaruvchilarini ajratib olish
x = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Ma'lumotlarni o'qish va test qilish uchun ajratib olish
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Entropiyaga asoslangan Decision Tree model yaratish
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

# Test qatorlariga qarshi bashoratlar aniqlash
y_pred = model.predict(X_test)

# Modelni aniqlilik baholash
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Meshgrid uchun katta qadamli vizualizatsiya
x_min, x_max = x.iloc[:, 0].min() - 1, x.iloc[:, 0].max() + 1
y_min, y_max = x.iloc[:, 1].min() - 1, x.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0), np.arange(y_min, y_max, 1.0))

# Qaror chizig'i
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, edgecolors='k', marker='o', s=100, linewidth=1)
plt.xlabel('Yosh')
plt.ylabel('Baholash qiymati')
plt.title("Decision Tree Classifier Qaror Chizig'i")
plt.show()

# Confusion Matrixni vizualizatsiya qilish
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Sotilmagan', 'Sotilgan'], yticklabels=['Sotilmagan', 'Sotilgan'])
plt.xlabel('Bashorat qilingan label')
plt.ylabel('Haqiqiy label')
plt.title('Confusion Matrix')
plt.show()
