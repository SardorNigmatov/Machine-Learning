import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ma'lumotlarni yuklash
df = pd.read_csv("./User_Data.csv")

# DataFrame ning bir nechta birinchi qatorlarini ko'rsatish
print(df.head())

# X va y ni aniqlash
x = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Ma'lumotlarni o'qish va sinov qismi uchun bo'lib ajratish
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Random Forest modelni yaratish va o'qitish
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Bashoratlarni aniqlash
y_pred = model.predict(X_test)

# Dastlabki aniqlikni hisoblash
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix ni chizish
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Bashorat')
plt.ylabel('Haqiqiy qiymat')
plt.show()
