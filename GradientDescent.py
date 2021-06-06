import math
import matplotlib.pyplot as plt
import pylab
import numpy as np
import random
from enum import Enum

n = 2  # размерность задачи
fragmentationCoef = 0.5  # коэффициент, на который домножается шаг при дроблении
fragmentationLimitCoef = 0.2  # коэффициент в условии выбора шага при дроблении


class StepFindingMethod(Enum):
    constantStep = 0
    stepFragmentation = 1

Q = [[2, 1], [1, 2]]
b = [0, 0]

# Q = [[2, 1], [1, 2]]
# b = [10, -1]

# Q = [[200000, 1], [1, 2]]
# b = [10, -1]


x_start = [40, -35]
e = 1e-5
alpha_start = 10
stepFindingMethod = StepFindingMethod.stepFragmentation


def f(x):
    sum = 0
    for i in range(n):
        for j in range(n):
            sum += 0.5 * Q[i][j] * x[i] * x[j]
    for i in range(n):
        sum += b[i] * x[i]
    return sum


def grad_f(x):
    result = []
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += Q[i][j] * x[j]
        sum += b[i]
        result.append(sum)
    return result


def norma(x):
    summa = 0
    for i in range(n):
        summa += x[i] ** 2
    return math.sqrt(summa)


x = []
alpha_arr = []


def substractPoints(x1, x2):
    result = []
    for i in range(len(x1)):
        result.append(x1[i] - x2[i])
    return result


# Метод градиентного спуска
def GradientDescent(x_start, alpha_start, epsilon, step_finding_method):
    x.append(x_start)
    alpha = alpha_start
    while True:

        # находим шаг согласно выбранному методу
        if step_finding_method == StepFindingMethod.stepFragmentation:
            if len(x) % 10 == 0:
                # периодически начинаем дробление заново
                alpha = alpha_start
            while True:
                # дробим шаг, пока он не удовлетворяет условию
                x_new = []
                for i in range(n):
                    x_new.append(x[-1][i] - alpha * grad_f(x[-1])[i])
                # проверяем шаг на соответствие условию релаксации
                if f(x_new) <= f(x[-1]) - fragmentationLimitCoef * alpha * (norma(grad_f(x[-1])) ** 2):
                    break
                alpha *= fragmentationCoef
        alpha_arr.append(alpha)

        # вычисляем следующее приближение
        x_new = []
        for i in range(n):
            x_new.append(x[-1][i] - alpha * grad_f(x[-1])[i])
        x.append(x_new)

        # условие остановки
        if norma(substractPoints(x[-1], x[-2])) < epsilon:
            x.pop()
            break
    print("Minimum point: ", x[-1], " found in ", len(x) - 1, " iterations")
    return len(x) - 1


minimum_found = True
try:
    GradientDescent(x_start, alpha_start, e, stepFindingMethod)
except OverflowError:
    print("Minimum not found")
    minimum_found = False

if n == 2 and minimum_found:
    # построение графиков
    f_arr = []
    for i in range(len(x)):
        f_arr.append(f(x[i]))

    x1 = []
    x2 = []
    for i in range(len(x)):
        x1.append(x[i][0])
        x2.append(x[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X1 = np.arange(min(x1) - 0.1 * (max(x1) - min(x1)), max(x1) + 0.1 * (max(x1) - min(x1)),
                   (max(x1) - min(x1)) * 0.001)
    X2 = np.arange(min(x2) - 0.1 * (max(x2) - min(x2)), max(x2) + 0.1 * (max(x2) - min(x2)),
                   (max(x2) - min(x2)) * 0.001)
    X1, X2 = np.meshgrid(X1, X2)
    Z = f([X1, X2])
    ax.plot_surface(X1, X2, f([X1, X2]), alpha=0.5)
    ax.plot(x1, x2, f_arr, color="black")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("График функции и траектория метода")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x1, x2, color="black")
    f_arr = list(set(f_arr))
    f_arr.sort()
    levels = pylab.contour(X1, X2, Z, f_arr)
    ax.contour(levels)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Линии уровня и траектория метода")
    plt.show()

# Исследование зависимости количества итераций от n и k
# n_range = [2, 3, 4, 5]
# for n in n_range:
#     successful = 0
#     k_arr = []
#     iter_arr = []
#     for k in range(200000):
#         if successful >= 1000:
#             break
#         x = []
#         alpha_arr = []
#         Q = []
#         b = []
#         for i in range(n):
#             string = []
#             for j in range(n):
#                 if i > j:
#                     string.append(Q[j][i])
#                 else:
#                     if i == j:
#                         string.append(random.uniform(1, 100))
#                     else:
#                         string.append(random.uniform(-100, 100))
#             Q.append(string)
#         for i in range(n):
#             b.append(random.uniform(-100, 100))
#         eigenvalues = np.linalg.eig(np.array(Q))[0]
#         tmp = []
#         for eigenvalue in eigenvalues:
#             if eigenvalue != 0:
#                 tmp.append(abs(eigenvalues))
#         eigenvalues = tmp
#         k_arr.append(max(eigenvalues) / min(eigenvalues))
#         x_start = []
#         for i in range(n):
#             x_start.append(random.randint(-100, 100))
#         try:
#             iter_arr.append(GradientDescent(x_start, alpha_start, e, stepFindingMethod))
#             successful += 1
#         except OverflowError:
#             k_arr.pop()
#         fig = plt.figure()
#         ax = fig.add_subplot()
#         ax.scatter(k_arr, iter_arr, color="black")
#         plt.xlabel("k")
#         plt.ylabel("iterations")
#         plt.title("Зависимость количества итераций от k, n = " + str(n))
#         z = np.polyfit(k_arr, iter_arr, 1)
#         p = np.poly1d(z)
#         pylab.plot(k_arr, p(k_arr))
#         print("i=%.6fk + %.6f" % (z[0], z[1]))
#         plt.show()
