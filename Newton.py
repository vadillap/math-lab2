import numpy as np


class F:
    def __init__(self, params):
        self.params = params

    def f(self, x):
        return self.params["f"](x)

    def dx1(self, x):
        return self.params["dx1"](x)

    def dx2(self, x):
        return self.params["dx2"](x)

    def dx1dx2(self, x):
        return self.params["dx1dx2"](x)

    def d2x1(self, x):
        return self.params["d2x1"](x)

    def d2x2(self, x):
        return self.params["d2x2"](x)

    def grad(self, x):
        return [
            self.dx1(x),
            self.dx2(x),
        ]

    # матрица Гессе, по сути аналог 2 производной
    def hesse(self, x):
        return [
            [self.d2x1(x), self.dx1dx2(x)],
            [self.dx1dx2(x), self.d2x2(x)],
        ]


def newton(x_0, f, epsilon):
    x = [x_0]
    while True:
        x_prev = x[-1]
        d1, d2 = f.grad(x_prev), f.hesse(x_prev)

        # основная формула метода ньютона
        x.append(x_prev - np.matmul(d1, np.linalg.inv(d2)))

        # проверяем условие останова
        if np.linalg.norm(x[-1] - x[-2]) < epsilon:
            x.pop()
            break

    return x


f1 = F({
    "f": lambda x: x[0] ** 2 + x[0] * x[1] + x[1] ** 2,
    "dx1": lambda x: 2 * x[0] + x[1],
    "dx2": lambda x: x[0] + 2 * x[1],
    "d2x1": lambda x: 2,
    "d2x2": lambda x: 2,
    "dx1dx2": lambda x: 1,
})

f2 = F({
    "f": lambda x: x[0] ** 2 + x[0] * x[1] + x[1] ** 2 + 10 * x[0] - x[1],
    "dx1": lambda x: 2 * x[0] + x[1] + 10,
    "dx2": lambda x: x[0] + 2 * x[1] - 1,
    "d2x1": lambda x: 2,
    "d2x2": lambda x: 2,
    "dx1dx2": lambda x: 1,
})

f3 = F({
    "f": lambda x: 100000 * x[0] ** 2 + x[0] * x[1] + x[1] ** 2 + 10 * x[0] - x[1],
    "dx1": lambda x: 200000 * x[0] + x[1] + 10,
    "dx2": lambda x: x[0] + 2 * x[1] - 1,
    "d2x1": lambda x: 200000,
    "d2x2": lambda x: 2,
    "dx1dx2": lambda x: 1,
})

x = newton([10, 10], f3, 1e-7)
print(x)
print("Minimum point:", x[-1], "found in", len(x) - 1, "iterations")
