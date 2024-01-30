# #Chiziqli regressiya

# import matplotlib.pyplot as plt

# # y = 2x + 1   biz tanlab olgan chiziqlii fuksiya
# x = [-1, 0, 1, 2, 3, 4, 5, 6]  # x qiymatlarimiz
# y = [-1, 1, 3, 5, 7, 9, 11, 13] # y qiymatlarimiz
# w = 0 # og'irlik
# b = 0 # bias

# lr = 0.001  # o'qitish  qadami

# epsilon = float(input("Epsilon qiymat:"))  # epsilon aniqlik

# # modelizmiz
# def predict(x, w, b):
#     return w * x + b

# # o'rtacha kvadratik xatolik
# def loss(x, y, w, b):
#     total_error = 0
#     for i in range(len(x)):
#         total_error += (y[i] - predict(x[i], w, b)) ** 2
#     return total_error / len(x)


# # Gradient decent algoritm yordamida  og'irlik va bias ni qiymatini yanilayapmiz
# def update_weights(x, y, w, b, lr):
#     w_deriv = 0
#     b_deriv = 0
#     for i in range(len(x)):
#         w_deriv += -2 * x[i] * (y[i] - predict(x[i], w, b)) # w ni  hisoblab olyapmiz
#         b_deriv += -2 * (y[i] - predict(x[i], w, b)) # biasni hisoblab olyapmiz
#     w -= (w_deriv / float(len(x))) * lr  # w ni qiymatini yangilab olyapmiz
#     b -= (b_deriv / float(len(x))) * lr # bias ni qiymatini yangilab olyapmiz
#     return w, b


# # O'qitish
# losses = []
# weights = []
# epoch = 0
# while True:
#     w, b = update_weights(x, y, w, b, lr) # bias va w ni qiymatini hisoblab olyapmiz
#     current_loss = loss(x, y, w, b)
#     losses.append(current_loss) # loss ni yig'ib ketyapmiz
#     weights.append(w) # w ni yig'ib ketyapmiz
#     epoch += 1   # epoch larni sanayapmiz
#     if current_loss < epsilon: # agar loss   epsilondan kichkina bo'lsa  siklni to'xtatyapmiz
#         break


# print("W ----------  Loss")
# for i in range(len(weights)):
#     if i % 10 == 0:
#         print(f"""{weights[i]} ---> {losses[i]}""")

# # Loss ni w ga bog'liq grafigi chizyapmiz
# plt.plot(weights,losses)
# plt.ylabel('Loss')
# plt.xlabel('w')
# plt.show()

# # Lossning epoch larga bog'liq grafigi  chizyapmiz
# plt.plot(list(range(epoch)),losses)
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.show()


# # Yankuniy natijamiz
# print("Og'irlik:", w)
# print("Bias:", b)
# print("Epochlar soni:", epoch)
# print("Bashorat:(x=7) bo'lgandagi",predict(7,w,b))



import matplotlib.pyplot as plt
class SimpleLinearRegression:
    """
    Oddiy logistik regressiya
    
    """
    def __init__(self):
        self.w = 0
        self.b = 0
    
    def predict(self,x):
        return self.w * x + self.b
    
    def loss(self,x,y):
        """Xatolik funksiyasi

        Args:
            x (_type_): erkli o'zgaruvchi
            y (_type_): erksiz o'zgaruvchi

        Returns:
            Umumiy xatolikni
        """
        total_error = sum((y[i]-self.predict(x[i]))**2 for i in range(len(x)))
        return total_error / len(x)
    
    def update_weights(self,x,y,lr):
        """Weights va biasni gradient descent yordamida qiymtini yangilash

        Args:
            x (_type_):  erkli o'zgaruvchi
            y (_type_): erksiz o'zgaruvchi
            lr (_type_): o'qitish qadami
        """
        w_deriv = sum(-2 * x[i] * (y[i] - self.predict(x[i])) for i in range(len(x)))
        b_deriv = sum(-2 * (y[i] - self.predict(x[i])) for i in range(len(x)))
        self.w -= (w_deriv / float(len(x))) * lr
        self.b -= (b_deriv / float(len(x))) * lr

    def train(self,x,y,lr,epsilon):
        """Modelni o'qitish funksiyasi

        Args:
            x (_type_): erkli o'zgaruvchi
            y (_type_): erksiz o'zgaruvchi
            lr (_type_): o'qitish qadami
            epsilon (_type_): aniqlik

        Returns:
            _type_: weights, losses. epoch
        """
        losses = []
        weights = []
        epoch = 0
        
        while True:
            self.update_weights(x,y,lr)
            current_loss = self.loss(x,y)
            losses.append(current_loss)
            weights.append(self.w)
            epoch += 1
            
            if current_loss < epsilon:
                break
            
        return weights, losses, epoch
    
linear = SimpleLinearRegression()

x_data = [-1, 0, 1, 2, 3, 4, 5, 6]
y_data = [-1, 1, 3, 5, 7, 9, 11, 13]

epsilon_value = float(input("Epsilon value:"))
weights, losses, epochs = linear.train(x_data,y_data,lr=0.001,epsilon=epsilon_value)

plt.plot(weights,losses)
plt.ylabel("Loss")
plt.xlabel("Weights")
plt.show()

plt.plot(list(range(epochs)), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


print("Weight:", linear.w)
print("Bias:", linear.b)
print("Number of epochs:", epochs)
print("Prediction for x=7:", linear.predict(7))