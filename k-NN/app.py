import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ma'lumotlarni yuklab olish
data = pd.read_csv('./User_Data.csv')

# Datasetning ma'lumotlari va natijalar
X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

# Ma'lumotlarni trenirovkasi va test qismiga bo'lib ajratib olish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN modelini yaratib, ma'lumotlarni o'qitish
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Test ma'lumotlarini o'qib, natijalarni bashorat qilish
y_pred = knn.predict(X_test)

# Aniqlik darajasi (accuracy)ni hisoblash
accuracy = accuracy_score(y_test, y_pred)
print(f"Aniqlik darajasi: {accuracy}")

# Confusion matrixni hisoblash
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Confusion matrixni vizualizatsiya qilish
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=data['Purchased'].unique(), yticklabels=data['Purchased'].unique())
plt.title('Confusion matrix')
plt.xlabel('Bashorat qilingan natijalar')
plt.ylabel('Haqiqiy natijalar')
plt.show()
