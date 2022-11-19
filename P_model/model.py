import math as mt
from dataclasses import dataclass


@dataclass
class Perceptron:
    Data: list
    n: int
    weights: list
    threshold = .5
    m = 0
    weights_sum = 0

    def step(self, w) -> int:
        return 1 if w > self.threshold else 0

    def perceptron(self, n):
        for i, j in zip(self.Data[n][0], self.weights):
            self.weights_sum += i * j
        return self.step(self.weights_sum)

    def fun(self, w_sum, j) -> float:
        m = 0
        for i in range(self.n - 1):
            w_sum += self.Data[j][0][i] * self.weights[i]
            m = round(w_sum, 2)
        return m

    def fun2(self) -> list:
        c = []
        for j in range(self.n):
            m = self.fun(0, j)
            c.append(m)
        return c

    def fun3(self):
        c = self.fun2()
        for i in range(self.n):
            p = self.perceptron(i)
            self.Data[i].append(p)
            self.Data[i].append(c[i])
        return self.Data

    def Loss(self):
        Data = self.fun3()
        loss = 0
        for i in range(self.n):
            w_sum = Data[i][1]
            y = Data[i][-1]
            loss += round(-(w_sum * mt.log10(y) + (1 - w_sum) * mt.log10(1 - y)), 2)
        return Data,f"Loss = {loss / self.n}"


Data = [
    [(0.1, 0.5, 0.2), 1],
    [(0.2, 0.3, 0.1), 0],
    [(0.7, 0.4, 0.2), 1],
    [(0.1, 0.4, 0.3), 0]
]
weights = [0.4, 0.2, 0.6]
def perceptron(Data_,weights_):
    return Perceptron(Data_, len(Data_), weights_)


p = perceptron(Data,weights)
#print(p.Loss())