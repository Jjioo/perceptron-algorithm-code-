import numpy as np

X = [0.1, 0.5, 0.2]  # our data
weigths = np.random.rand(1, 3)[0]  # our weigths
threshold = 0.5  # our threshold


#########################
###Activation Function###
#########################

def step(weigths_sum):
    return 1 if weigths_sum > threshold else 0


#########################
####simple Perceptron####
#########################

def Perceptron():
    weigths_sum = 0
    for x, y in zip(X, weigths):
        weigths_sum += x * y
    print(weigths_sum)
    return step(weigths_sum)


p = Perceptron()
print(p)