# 50 yoshli erkak uchun sistolik qon bosim   120 mm - 140 mm simob ustuni bo'lsa normal
# diastolik 80 - 90  mm simob ustuni normal hisoblandi
# yurak urishlar soni 80 - 100  ta
# 1 - kasallik yo'q
# 0 - kasallik bor

# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier


# # Ma'lumotlarni olish
# data_set = pd.read_csv('data.csv')
# X = data_set.iloc[:, [1, 2, 3]].values
# Y = data_set.iloc[:, -1].values

# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X, Y)


# sistolik = data_set.iloc[:, [1]].values
# yurak = data_set.iloc[:, [3]].values

# # Joriy nuqtalarni chizish
# plt.scatter(sistolik, yurak, color='red', s=60, label='Joriy nuqtalar')

# # Boshorat qiymatini kiritish
# predict_value = tuple(map(int, input("Bashorat qiymatlarni kiriting: ").split()))

# # KNN modeli orqali natijani hisoblash
# y_pred = knn.predict([predict_value])

# # Boshorat nuqtasini chizish
# plt.scatter(predict_value[0], predict_value[2], color='blue', s=60, label='Boshorat nuqtasi')

# # Grafik so'rovnoma va o'zgaruvchilarni hisoblash
# plt.title("k-NN")
# plt.xlabel("Sistolik qon bosimi")
# plt.ylabel("Yurak urishlar soni")
# plt.legend()

# plt.show()

# print("Natija:", y_pred)





# # Decision Tree modeli
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X, Y)

# sistolik = data_set.iloc[:, [1]].values
# yurak = data_set.iloc[:, [3]].values

# # Joriy nuqtalarni chizish
# plt.scatter(sistolik, yurak, color='red', s=60, label='Joriy nuqtalar')

# # Boshorat qiymatini kiritish
# predict_value = tuple(map(int, input("Bashorat qiymatlarni kiriting: ").split()))

# # Decision Tree modeli orqali natijani hisoblash
# y_pred = decision_tree.predict([predict_value])

# # Boshorat nuqtasini chizish
# plt.scatter(predict_value[0], predict_value[2], color='blue', s=60, label='Boshorat nuqtasi')


# plt.title("Decision Tree")
# plt.xlabel("Sistolik qon bosimi")
# plt.ylabel("Yurak urishlar soni")
# plt.legend()

# plt.show()

# print("Natija:", y_pred)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# CSV faylini o'qish
data = pd.read_csv('data.csv')

# X va y larni ajratib olish
X = data.drop('sinf', axis=1)
y = data['sinf']

# Ma'lumotlarni trenirovka va test qismiga ajratib olish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Support Vector Machine (SVM) modelini yaratish va o'qitish
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Test ma'lumotlari uchun bashorat
y_pred = model.predict(X_test)

# Aniqlikni hisoblash
accuracy = accuracy_score(y_test, y_pred)
print(f'Test ma\'lumotlari uchun aniqlik: {accuracy:.4f}')


plt.scatter(data['Sistolik'], data['Diastolik'], c=data['sinf'], cmap='viridis')
plt.title('Nuqtalar')
plt.xlabel('Sistolik')
plt.ylabel('Diastolik')
plt.show()

# Konfuziya matricasini o'qish va ko'rsatish
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.ylabel('Haqiqiy natijalar')
plt.xlabel('Bashorat natijalari')
plt.title('Confusion Matrix')
plt.show()
