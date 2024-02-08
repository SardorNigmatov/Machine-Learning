# # Kvadratik model

# # y = x^2 - 3x + 2

# import matplotlib as plt

# x = [1, 2, 3, 4, 5, 6] # x ning qiymatlari
# y = [0, 0, 2, 6, 12, 20] # y ning qiymatlari
# w1 = 0
# w2 = 0
# b = 0
# lr = 0.001  # o'qitish qadami

# epsilon = float(input("Epsilon:"))

# # Function to predict y
# def predict(x, w1, w2, b):
#     return w1 * (x**2) + w2 * x + b

# # Loss funksiya o'rtacha kvadratik xatolikni hisoblab olyapmiz
# def loss(x, y, w1, w2, b):
#     total_error = 0
#     for i in range(len(x)):
#         total_error += (y[i] - predict(x[i], w1, w2, b)) ** 2
#     return total_error / len(x)

# # gredient decent  yordamida w1, w2, va bais larni yangilab olyapmiz
# def update_weights(x, y, w1, w2, b, lr):
#     w1_deriv = 0
#     w2_deriv = 0
#     b_deriv = 0
#     for i in range(len(x)):
#         w1_deriv += -2 * (x[i]**2) * (y[i] - predict(x[i], w1, w2, b)) # w1 hisoblayapmiz
#         w2_deriv += -2 * x[i] * (y[i] - predict(x[i], w1, w2, b)) # w2 hisoblayapmiz
#         b_deriv += -2 * (y[i] - predict(x[i], w1, w2, b)) # bais hisoblayapmiz
#     w1 -= (w1_deriv / float(len(x))) * lr # w1 yangilab olyamiz
#     w2 -= (w2_deriv / float(len(x))) * lr # w2  yangilab olyapmiz
#     b -= (b_deriv / float(len(x))) * lr # bais ni yangilab olyapmiz
#     return w1, w2, b

# # Training loop
# losses = []
# weights1 = []
# weights2 = []
# epoch = 0

# while True:
#     w1, w2, b = update_weights(x, y, w1, w2, b, lr)
#     current_loss = loss(x, y, w1, w2, b)
#     losses.append(current_loss)
#     weights1.append(w1)
#     weights2.append(w2)
#     epoch += 1 # epochlar soninni sanayapmiz
#     if current_loss < epsilon: # loss imiz epsilondan kichkina bo'lsa  siklni to'xtatyapmiz
#         break



# for i in range(len(weights1)):
#     if i % 800 == 0:
#         print("Weight1: ", weights1[i], "Weight2: ", weights2[i], " Loss: ", losses[i])

# # Loss ni w1  ga bog'liq grafigin chizib olyapmiz
# plt.plot(weights1,losses)
# plt.ylabel('Loss')
# plt.xlabel('w1')
# plt.show()


# # Loss ni w2 ga bog'liq grafigini chizib  olyapmiz
# plt.plot(weights2,losses)
# plt.ylabel('Loss')
# plt.xlabel('w2')
# plt.show()

# # Lossning epoch larga bog'liq grafigi chizib olyapmiz
# plt.plot(list(range(epoch)),losses)
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.show()

# print("w1:", w1)
# print("Fw2:", w2)
# print("bias:", b)
# print("Epochlar soni:", epoch)
# print("Bashorat:(x = 7) bo'lgandagi:",predict(7,w1,w2,b))



import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt

class QuadraticModel:
    def __init__(self, x, y, lr=0.2):
        self.x = x
        self.y = y
        self.lr = lr
        self.w1 = 0
        self.w2 = 0
        self.b = 0
        self.weight1 = []
        self.weight2 = []
        self.losses = []

    def predict(self, x):
        if isinstance(x,list):
             return [ self.w1 * (i**2) + self.w2 * i + self.b for i in x]
        else:
             return self.w1 * (x**2) + self.w2 * x + self.b

    def loss(self):
        total_error = sum((self.y[i] - self.predict(self.x[i]))**2 for i in range(len(self.x)))
        return total_error / len(self.x)

    def update_weights(self):
        w1_deriv = 0
        w2_deriv = 0
        b_deriv = 0
        for i in range(len(self.x)):
            w1_deriv += -2 * (self.x[i]**2) * (self.y[i] - self.predict(self.x[i]))
            w2_deriv += -2 * self.x[i] * (self.y[i] - self.predict(self.x[i]))
            b_deriv += -2 * (self.y[i] - self.predict(self.x[i]))

        self.w1 -= (w1_deriv / float(len(self.x))) * self.lr
        self.w2 -= (w2_deriv / float(len(self.x))) * self.lr
        self.b -= (b_deriv / float(len(self.x))) * self.lr
        self.weight1.append(self.w1)
        self.weight2.append(self.w2)
    

    def train(self, epsilon=0.001,iteratsion=1000):
        epoch = 0

        while epoch < iteratsion:
            self.update_weights()
            current_loss = self.loss()
            self.losses.append(current_loss)
            epoch += 1
            if current_loss < epsilon:
                break

        return epoch

    def plot_loss_vs_weights(self, weights, xlabel):
        plt.plot(weights, self.losses)
        plt.ylabel('Loss')
        plt.xlabel(xlabel)
        plt.show()

if __name__ == "__main__":
    x_values = pd.read_csv("x_data.csv").iloc[:,0].tolist() #[1, 2, 3, 4, 5, 6] 
    y_values = pd.read_csv("y_data.csv").iloc[:,0].tolist() #[0, 0, 2, 6, 12, 20] 
 
    quadratic_model = QuadraticModel(x_values, y_values)
    
    epsilon_value = float(input("Epsilon: "))
    num_epochs = quadratic_model.train(epsilon=epsilon_value,iteratsion=70000)

    print("Losses count",len(quadratic_model.losses))
    print("w1:", quadratic_model.w1)
    print("w2:", quadratic_model.w2)
    print("bias:", quadratic_model.b)
    print("Epochlar soni:", num_epochs)
    print("Bashorat:(x = 0.5) bo'lgandagi:", quadratic_model.predict(0.5))

    quadratic_model.plot_loss_vs_weights(quadratic_model.weight1, 'w1')
    quadratic_model.plot_loss_vs_weights(quadratic_model.weight2, 'w2')
    plt.plot(list(range(num_epochs)),quadratic_model.losses)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()
    
    x_range = np.linspace(min(quadratic_model.x), max(quadratic_model.x), 200)
    y_predicted = quadratic_model.predict(x_range)

    plt.scatter(quadratic_model.x, quadratic_model.y, label='Data Points')
    plt.plot(x_range, y_predicted, label='Quadratic Model', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    
    
    
    
#Kutubxonadan foydalanib yozilgan model

import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Ma'lumotlarni yuklab olish
x_values = pd.read_csv("x_data.csv").iloc[:, 0].values.reshape(-1, 1)
y_values = pd.read_csv("y_data.csv").iloc[:, 0].values

# Modelni yaratish
degree = 2  # Kvadratik model uchun
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Modelni o'rganish
model.fit(x_values, y_values)

# Modelni ko'rish uchun diapazon belgilash
x_range = np.linspace(min(x_values), max(x_values), 100).reshape(-1, 1)
y_predicted = model.predict(x_range)

# Bashorat
x_test = np.array([0.5, 0]).reshape(-1, 1)  # X qiymatlarini 2D arrayga o'tkazamiz
predictions = model.predict(x_test)
print("Bashorat:", predictions[0])

# Ma'lumotlarni va o'rgangan modelni chizish
plt.scatter(x_values, y_values, label='Data Points')
plt.plot(x_range, y_predicted, label='Quadratic Model', color='red')
plt.scatter([0.5, 0], predictions, color='blue', label='Bashorat')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
