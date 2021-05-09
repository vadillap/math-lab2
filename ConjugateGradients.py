import math
import matplotlib.pyplot as plt
import pylab
import numpy as np
from enum import Enum


class StepFindingMethod(Enum):
    goldenSection = 0
    fibonacci = 1


def f(x1, x2):
    return 2 * x1 ** 2 + x2 ** 2 - x1 * x2 + x1 - x2


def df_dx1(x1, x2):
    return 4 * x1 - x2 + 1


def df_dx2(x1, x2):
    return 2 * x2 - x1 - 1


def g(x1, x2, alpha, p):
    return f(x1 + alpha * p[0], x2 + alpha * p[1])


def norma(x1, x2):
    return math.sqrt(x1 ** 2 + x2 ** 2)


def scalar_multiply(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += x1[i] * x2[i]
    return sum


# Число Фибоначчи
def Fibonacci(n):
    return int(((1 + math.sqrt(5)) ** n - (1 - math.sqrt(5)) ** n) / (2 ** n * math.sqrt(5)))


# Метод Фибоначчи
def Fibonacci_Method(x_min, x_max, number_of_iterations, x1, x2, p, iteration=0):
    if iteration == number_of_iterations:
        return (x_max + x_min) / 2
    x_lhs = x_min + (((x_max - x_min) * Fibonacci(number_of_iterations - iteration - 1)) / Fibonacci(
        number_of_iterations - iteration + 1))
    x_rhs = x_min + (((x_max - x_min) * Fibonacci(number_of_iterations - iteration)) / Fibonacci(
        number_of_iterations - iteration + 1))
    if g(x1, x2, x_lhs, p) > g(x1, x2, x_rhs, p):
        return Fibonacci_Method(x_lhs, x_max, number_of_iterations, x1, x2, p, iteration + 1)
    else:
        return Fibonacci_Method(x_min, x_rhs, number_of_iterations, x1, x2, p, iteration + 1)


# Метод золотого сечения
def GoldenSectionMethod(x_min, x_max, epsilon, x1, x2, p):
    iteration = 1
    coefficient = (math.sqrt(5) - 1) / 2
    d = x_min + (x_max - x_min) * coefficient
    c = x_max - (x_max - x_min) * coefficient
    sc = g(x1, x2, c, p)
    sd = g(x1, x2, d, p)
    while (x_max - x_min) > epsilon:
        if sd > sc:
            x_max = d
            d = c
            c = x_max - (x_max - x_min) * coefficient
            sd = sc
            sc = g(x1, x2, c, p)
        else:
            x_min = c
            c = d
            d = x_min + (x_max - x_min) * coefficient
            sc = sd
            sd = g(x1, x2, d, p)
        iteration += 1
    return (x_max + x_min) / 2


x1 = []
x2 = []
alpha = []
p = []
beta = []


# Метод сопряженных градиентов
def ConjugateGradients(x1_start, x2_start, epsilon, step_finding_method):
    x1.append(x1_start)
    x2.append(x2_start)
    p.append([-df_dx1(x1[-1], x2[-1]), -df_dx2(x1[-1], x2[-1])])
    beta.append(0)
    restart_interval = 2
    k = 0
    while True:

        # находим шаг согласно выбранному методу одномерной оптимизации
        if step_finding_method == StepFindingMethod.goldenSection:
            alpha.append(GoldenSectionMethod(0, 1000000, epsilon, x1[-1], x2[-1], p[-1]))
        if step_finding_method == StepFindingMethod.fibonacci:
            alpha.append(Fibonacci_Method(0, 1000000, 100, x1[-1], x2[-1], p[-1]))

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


ConjugateGradients(-40.0, 50.0, 0.0001, StepFindingMethod.goldenSection)

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
f_arr.sort()
levels = pylab.contour(X1, X2, Z, f_arr)
plt.plot(x1, x2)
plt.clabel(levels)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Линии уровня и траектория метода")
plt.show()
