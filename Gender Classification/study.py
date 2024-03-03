import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Faylni o'qish
df = pd.read_csv("./odamlar.csv")

# Xususiyatlarni va yozuvlarni ajratib olish
features = df[['Age', 'Height(cm)', 'Weight(kg)', 'Income(USD)']]
labels = df['Gender']

# To'g'ridan-to'g'ri hariflarini raqamga aylantirish
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Jinslarni ikkita guruhga ajratish: 1 - 'Male', 0 - 'Female'
labels_binary = labels.copy()
labels_binary[labels_binary != label_encoder.transform(['male'])[0]] = 0

# Ma'lumotlarni o'qish va test va trening qismlarga bo'lish
X_train, X_test, y_train, y_test = train_test_split(features, labels_binary, test_size=0.25, random_state=42)

# k-NN klassifikatori yaratish
k = 7
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Klassifikatorni o'qitish
knn_classifier.fit(X_train, y_train)

# Test to'plamida bashorat qilish
predictions = knn_classifier.predict(X_test)

# Aniqlovchi natijalarni hisoblash
accuracy = accuracy_score(y_test, predictions)
print(f"Aniqlik: {accuracy * 100:.2f}%")


# Ikki guruh uchun aniqlovchi matritsa yaratish ('Male' vs. 'Female' uchun)
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['female', 'male'], yticklabels=['female', 'male'])
plt.xlabel("Bashorat qilingan qatori")
plt.ylabel("To'g'ri qatori")
plt.title("Bashorat Qilish Matritsasi ('Male' vs. 'Female')")
plt.show()

# Ikki guruh uchun aniqlovchi hisobot yaratish
class_report = classification_report(y_test, predictions, target_names=['female', 'male'], zero_division=1)
print("Aniqlovchi Hisobot:\n", class_report)

# Actual vs Predicted label uchun scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['Age'], y=X_test['Height(cm)'], hue=label_encoder.inverse_transform(predictions), palette='viridis')
plt.title("Bashorat qilingan va aniqlovchi label larining scatter plot i ('Male' vs. 'Female')")
plt.xlabel("Yosh")
plt.ylabel("Balandlik (sm)")
plt.legend(title="Bashorat Qilingan Jins")
plt.show()
