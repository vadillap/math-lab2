import math


def f(x1, x2):
    return x1 ** 2 - x2 ** 2


def df_dx1(x1, x2):
    return 2 * x1 - x2 ** 2


def df_dx2(x1, x2):
    return x1 ** 2 - 2 * x2


def g(x1, x2, alpha):
    return f(x1 - alpha * df_dx1(x1, x2), x2 - alpha * df_dx2(x1, x2))


def norma(x1, x2):
    return math.sqrt(x1 ** 2 + x2 ** 2)


# Число Фибоначчи
def Fibonacci(n):
    return int(((1 + math.sqrt(5)) ** n - (1 - math.sqrt(5)) ** n) / (2 ** n * math.sqrt(5)))


# Метод Фибоначчи
def Fibonacci_Method(x_min, x_max, number_of_iterations, x1, x2, iteration=0):
    if iteration == number_of_iterations:
        return (x_max - x_min) / 2 + x_min
    x_lhs = x_min + (((x_max - x_min) * Fibonacci(number_of_iterations - iteration - 1)) / Fibonacci(
        number_of_iterations - iteration + 1))
    x_rhs = x_min + (((x_max - x_min) * Fibonacci(number_of_iterations - iteration)) / Fibonacci(
        number_of_iterations - iteration + 1))
    if g(x1, x2, x_lhs) > g(x1, x2, x_rhs):
        return Fibonacci_Method(x_lhs, x_max, number_of_iterations, x1, x2, iteration + 1)
    else:
        return Fibonacci_Method(x_min, x_rhs, number_of_iterations, x1, x2, iteration + 1)


# Метод золотого сечения
def GoldenSectionMethod(x_min, x_max, epsilon, x1, x2):
    iteration = 1
    coefficient = (math.sqrt(5) - 1) / 2
    d = x_min + (x_max - x_min) * coefficient
    c = x_max - (x_max - x_min) * coefficient
    sc = g(x1, x2, c)
    sd = g(x1, x2, d)
    while (x_max - x_min) > epsilon:
        if sd > sc:
            x_max = d
            d = c
            c = x_max - (x_max - x_min) * coefficient
            sd = sc
            sc = g(x1, x2, c)
        else:
            x_min = c
            c = d
            d = x_min + (x_max - x_min) * coefficient
            sc = sd
            sd = g(x1, x2, d)
        iteration += 1
    return (x_max - x_min) / 2 + x_min


# Метод наискорейшего спуска
def SteepestDescent(x1_start, x2_start, epsilon):
    # n = x.length
    x1 = []
    x2 = []
    alpha_arr = []
    x1.append(x1_start)
    x2.append(x2_start)
    while True:
        alpha_arr.append(Fibonacci_Method(0, 100, 26, x1[len(x1) - 1], x2[len(x2) - 1]))
        # alpha_arr.append(GoldenSectionMethod(0, 100, epsilon, x1[len(x1) - 1], x2[len(x2) - 1]))
        x1.append(x1[len(x1) - 1] - alpha_arr[len(alpha_arr) - 1] * df_dx1(x1[len(x1) - 1], x2[len(x2) - 1]))
        x2.append(x2[len(x2) - 1] - alpha_arr[len(alpha_arr) - 1] * df_dx2(x1[len(x1) - 1], x2[len(x2) - 1]))
        if norma(x1[len(x1) - 1] - x1[len(x1) - 2], x2[len(x2) - 1] - x2[len(x2) - 2]) < epsilon:
            break
    print(x1[len(x1) - 1], " ", x2[len(x2) - 1])
