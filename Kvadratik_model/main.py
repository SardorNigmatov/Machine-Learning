# Kvadratik model

# y = x^2 - 3x + 2

x = [1, 2, 3, 4, 5, 6] # x ning qiymatlari
y = [0, 0, 2, 6, 12, 20] # y ning qiymatlari
w1 = 0
w2 = 0
b = 0
lr = 0.001  # o'qitish qadami

epsilon = float(input("Epsilon:"))

# Function to predict y
def predict(x, w1, w2, b):
    return w1 * (x**2) + w2 * x + b

# Loss funksiya o'rtacha kvadratik xatolikni hisoblab olyapmiz
def loss(x, y, w1, w2, b):
    total_error = 0
    for i in range(len(x)):
        total_error += (y[i] - predict(x[i], w1, w2, b)) ** 2
    return total_error / len(x)

# gredient decent  yordamida w1, w2, va bais larni yangilab olyapmiz
def update_weights(x, y, w1, w2, b, lr):
    w1_deriv = 0
    w2_deriv = 0
    b_deriv = 0
    for i in range(len(x)):
        w1_deriv += -2 * (x[i]**2) * (y[i] - predict(x[i], w1, w2, b)) # w1 hisoblayapmiz
        w2_deriv += -2 * x[i] * (y[i] - predict(x[i], w1, w2, b)) # w2 hisoblayapmiz
        b_deriv += -2 * (y[i] - predict(x[i], w1, w2, b)) # bais hisoblayapmiz
    w1 -= (w1_deriv / float(len(x))) * lr # w1 yangilab olyamiz
    w2 -= (w2_deriv / float(len(x))) * lr # w2  yangilab olyapmiz
    b -= (b_deriv / float(len(x))) * lr # bais ni yangilab olyapmiz
    return w1, w2, b

# Training loop
losses = []
weights1 = []
weights2 = []
epoch = 0

while True:
    w1, w2, b = update_weights(x, y, w1, w2, b, lr)
    current_loss = loss(x, y, w1, w2, b)
    losses.append(current_loss)
    weights1.append(w1)
    weights2.append(w2)
    epoch += 1 # epochlar soninni sanayapmiz
    if current_loss < epsilon: # loss imiz epsilondan kichkina bo'lsa  siklni to'xtatyapmiz
        break



for i in range(len(weights1)):
    if i % 800 == 0:
        print("Weight1: ", weights1[i], "Weight2: ", weights2[i], " Loss: ", losses[i])

# Loss ni w1  ga bog'liq grafigin chizib olyapmiz
plt.plot(weights1,losses)
plt.ylabel('Loss')
plt.xlabel('w1')
plt.show()


# Loss ni w2 ga bog'liq grafigini chizib  olyapmiz
plt.plot(weights2,losses)
plt.ylabel('Loss')
plt.xlabel('w2')
plt.show()

# Lossning epoch larga bog'liq grafigi chizib olyapmiz
plt.plot(list(range(epoch)),losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

print("w1:", w1)
print("Fw2:", w2)
print("bias:", b)
print("Epochlar soni:", epoch)
print("Bashorat:(x = 7) bo'lgandagi:",predict(7,w1,w2,b))