from model import perceptron
import numpy as np
Data = [
    [(0.1, 0.5, 0.2), 1],
    [(0.2, 0.3, 0.1), 0],
    [(0.7, 0.4, 0.2), 1],
    [(0.1, 0.4, 0.3), 0]
]

#w = np.random.rand(1, 3)[0]
w=[0.4, 0.2, 0.6]
p = perceptron(Data,w)
print(p.Loss()[-1])
print("------DATA-----| actually target|    our target|    |w_sum")
for i in p.Loss()[0]:
    print(i[0],"\t\t\t  ",i[1],"\t\t\t ",i[2],"    ",i[-1],sep='|')