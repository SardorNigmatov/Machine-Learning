import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc

# Ma'lumotlarni yuklash
df = pd.read_csv("./User_Data.csv")

# Datasetning bir nechta birinchi qatorlarini ko'rsatish
print(df.head())

# X va Y ni tanlash
x = df.iloc[:, [2, 3]].values
y = df.iloc[:, 4].values

# X va Y ni ko'rsatish
print("X:", x)
print("Y:", y)

# Datasetni o'qish va sinov qatorlarga bo'lish
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# X larni standartlash
standard_x = StandardScaler()
x_train = standard_x.fit_transform(x_train)
x_test = standard_x.transform(x_test)

# Logistik regressiya modelini yaratish
model = LogisticRegression()

# Modelni o'qitish
model.fit(x_train, y_train)

# Test qatorida bashorat qilish
y_pred = model.predict(x_test)

# Bashorat qiymatlarini ko'rsatish
print("y_pred:", y_pred)

# Confusion matrixni hisoblash
cfm = confusion_matrix(y_test, y_pred)

# Confusion matrixni ko'rsatish
print("Confusion Matrix:")
print(cfm)

# Qo'shimcha metrikalarni hisoblash
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Qo'shimcha metrikalarni ko'rsatish
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Confusion matrixni chizish
sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC kurvani chizish
y_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
