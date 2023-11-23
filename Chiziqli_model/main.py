#Chiziqli regressiya

import matplotlib.pyplot as plt

# y = 2x + 1   biz tanlab olgan chiziqlii fuksiya
x = [-1, 0, 1, 2, 3, 4, 5, 6]  # x qiymatlarimiz
y = [-1, 1, 3, 5, 7, 9, 11, 13] # y qiymatlarimiz
w = 0 # og'irlik
b = 0 # bias

lr = 0.001  # o'qitish  qadami

epsilon = float(input("Epsilon qiymat:"))  # epsilon aniqlik

# modelizmiz
def predict(x, w, b):
    return w * x + b

# o'rtacha kvadratik xatolik
def loss(x, y, w, b):
    total_error = 0
    for i in range(len(x)):
        total_error += (y[i] - predict(x[i], w, b)) ** 2
    return total_error / len(x)


# Gradient decent algoritm yordamida  og'irlik va bias ni qiymatini yanilayapmiz
def update_weights(x, y, w, b, lr):
    w_deriv = 0
    b_deriv = 0
    for i in range(len(x)):
        w_deriv += -2 * x[i] * (y[i] - predict(x[i], w, b)) # w ni  hisoblab olyapmiz
        b_deriv += -2 * (y[i] - predict(x[i], w, b)) # biasni hisoblab olyapmiz
    w -= (w_deriv / float(len(x))) * lr  # w ni qiymatini yangilab olyapmiz
    b -= (b_deriv / float(len(x))) * lr # bias ni qiymatini yangilab olyapmiz
    return w, b


# O'qitish
losses = []
weights = []
epoch = 0
while True:
    w, b = update_weights(x, y, w, b, lr) # bias va w ni qiymatini hisoblab olyapmiz
    current_loss = loss(x, y, w, b)
    losses.append(current_loss) # loss ni yig'ib ketyapmiz
    weights.append(w) # w ni yig'ib ketyapmiz
    epoch += 1   # epoch larni sanayapmiz
    if current_loss < epsilon: # agar loss   epsilondan kichkina bo'lsa  siklni to'xtatyapmiz
        break


print("W ----------  Loss")
for i in range(len(weights)):
    if i % 10 == 0:
        print(f"""{weights[i]} ---> {losses[i]}""")

# Loss ni w ga bog'liq grafigi chizyapmiz
plt.plot(weights,losses)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()

# Lossning epoch larga bog'liq grafigi  chizyapmiz
plt.plot(list(range(epoch)),losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()


# Yankuniy natijamiz
print("Og'irlik:", w)
print("Bias:", b)
print("Epochlar soni:", epoch)
print("Bashorat:(x=7) bo'lgandagi",predict(7,w,b))
