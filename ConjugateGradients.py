import math
import matplotlib.pyplot as plt
import pylab
import numpy as np

Q = [[2, 1], [1, 2]]
b = [0, 0]

# Q = [[2, 1], [1, 2]]
# b = [10, -1]

# Q = [[200000, 1], [1, 2]]
# b = [10, -1]

x1_start = 100
x2_start = -100
e = 0.0001


def f(x1, x2):
    return 0.5 * (Q[0][0] * x1 ** 2 + Q[0][1] * x1 * x2 + Q[1][0] * x1 * x2 + Q[1][1] * x2 ** 2) + x1 * b[0] + x2 * b[1]


def df_dx1(x1, x2):
    return Q[0][0] * x1 + Q[0][1] * x2 + b[0]


def df_dx2(x1, x2):
    return Q[1][0] * x1 + Q[1][1] * x2 + b[1]


def g(x1, x2, alpha, p):
    return f(x1 + alpha * p[0], x2 + alpha * p[1])


def norma(x1, x2):
    return math.sqrt(x1 ** 2 + x2 ** 2)


def scalar_multiply(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += x1[i] * x2[i]
    return sum


x1 = []
x2 = []
alpha = []
p = []
beta = []


# вычисляем шаг согласно формуле для метода сопряженных направлений
def countAlpha(grad, p):
    if p[0] == 0 and p[1] == 0:
        return 0
    return scalar_multiply(grad, p) / (
            Q[0][0] * p[0] ** 2 + Q[0][1] * p[0] * p[1] + Q[1][0] * p[0] * p[1] + Q[1][1] * p[1] ** 2)


# Метод сопряженных градиентов
def ConjugateGradients(x1_start, x2_start, epsilon):
    x1.append(x1_start)
    x2.append(x2_start)
    p.append([-df_dx1(x1[-1], x2[-1]), -df_dx2(x1[-1], x2[-1])])
    beta.append(0)
    restart_interval = 2
    k = 0
    while True:

        alpha.append(countAlpha([-df_dx1(x1[-1], x2[-1]), -df_dx2(x1[-1], x2[-1])], p[-1]))

        # вычисляем следующее приближение
        x1.append(x1[-1] + alpha[-1] * p[-1][0])
        x2.append(x2[-1] + alpha[-1] * p[-1][1])

        # условие остановки
        if norma(x1[-1] - x1[-2], x2[-1] - x2[-2]) < epsilon:
            k += 1
            break

        if (k + 1) % restart_interval == 0:
            # через каждые restart_interval шагов делаем рестарт метода - обнуляем коэффициент beta
            beta.append(0)
        else:
            # вычисляем коэффициент beta
            beta.append((norma(df_dx1(x1[-1], x2[-1]), df_dx2(x1[-1], x2[-1])) ** 2) / scalar_multiply(
                [df_dx1(x1[-2], x2[-2]), df_dx2(x1[-2], x2[-2])], p[-1]))

        # Вычисляем следующее направление
        p.append([-df_dx1(x1[-1], x2[-1]) + p[-1][0] * beta[-1],
                  -df_dx2(x1[-1], x2[-1]) + p[-1][1] * beta[-1]])
        k += 1
    print("Minimum point: ", [x1[-1], x2[-1]], " found in ", k, " iterations")


ConjugateGradients(x1_start, x2_start, e)

# построение графиков
f_arr = []
for i in range(len(x1)):
    f_arr.append(f(x1[i], x2[i]))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X1 = np.arange(min(x1) - 0.1 * (max(x1) - min(x1)), max(x1) + 0.1 * (max(x1) - min(x1)), 0.1)
X2 = np.arange(min(x2) - 0.1 * (max(x2) - min(x2)), max(x2) + 0.1 * (max(x2) - min(x2)), 0.1)
X1, X2 = np.meshgrid(X1, X2)
Z = f(X1, X2)
ax.plot_surface(X1, X2, f(X1, X2), alpha=0.5)
ax.plot(x1, x2, f_arr, color="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("График функции и траектория метода")
plt.show()
f_arr = list(set(f_arr))
f_arr.sort()
levels = pylab.contour(X1, X2, Z, f_arr)
plt.plot(x1, x2)
plt.clabel(levels)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Линии уровня и траектория метода")
plt.show()
