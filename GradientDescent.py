import math
import matplotlib.pyplot as plt
import pylab
import numpy as np
from enum import Enum

n = 2
fragmentationCoef = 0.2
fragmentationLimitCoef = 0.1


class StepFindingMethod(Enum):
    constantStep = 0
    stepFragmentation = 1


def f(x):
    return 2 * x[0] ** 2 + x[1] ** 2 - x[0] * x[1] + x[0] - x[1]


def df_dx1(x1, x2):
    return 4 * x1 - x2 + 1


def df_dx2(x1, x2):
    return 2 * x2 - x1 - 1


def grad_f(x):
    return [4 * x[0] - x[1] + 1, 2 * x[1] - x[0] - 1]


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
        if step_finding_method == StepFindingMethod.stepFragmentation:
            if len(x) % 100 == 0:
                alpha = alpha_start
            while True:
                x_new = []
                for i in range(n):
                    x_new.append(x[-1][i] - alpha * grad_f(x[-1])[i])
                if f(x_new) <= f(x[-1]) - fragmentationLimitCoef * alpha * (norma(grad_f(x[-1])) ** 2):
                    break
                alpha *= fragmentationCoef
        alpha_arr.append(alpha)
        x_new = []
        for i in range(n):
            x_new.append(x[-1][i] - alpha * grad_f(x[-1])[i])
        x.append(x_new)
        if norma(substractPoints(x[-1], x[-2])) < epsilon:
            break
    print("Minimum point: ", x[-1], " found in ", len(x) - 1, " iterations")


GradientDescent([30, -10], 0.1, 0.0005, StepFindingMethod.constantStep)

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
X1 = np.arange(min(x1) - 0.1 * (max(x1) - min(x1)), max(x1) + 0.1 * (max(x1) - min(x1)), 0.1)
X2 = np.arange(min(x2) - 0.1 * (max(x2) - min(x2)), max(x2) + 0.1 * (max(x2) - min(x2)), 0.1)
X1, X2 = np.meshgrid(X1, X2)
Z = f([X1, X2])
ax.plot_surface(X1, X2, f([X1, X2]), alpha=0.5)
ax.plot(x1, x2, f_arr, color="black")
plt.show()
f_arr.sort()
levels = pylab.contour(X1, X2, Z, f_arr)
pylab.clabel(levels)
pylab.plot(x1, x2)
pylab.show()