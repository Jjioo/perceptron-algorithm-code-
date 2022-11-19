import math as mt
Data      = [
            [(0.1,0.5,0.2), 1], 
            [(0.2,0.3,0.1), 0],
            [(0.7,0.4,0.2), 1],
            [(0.1,0.4,0.3), 0]
          ]

n= len(Data)
weights = [0.4,0.2,0.6]
w= []
threshold =.5
def step(weights_sum):
  return 1 if weights_sum>threshold else 0
def Perceptron(i):
  weights_sum = 0
  for i,j in zip(Data[i][0],weights):
      weights_sum += i*j
  return step(weights_sum)

c= []
s=0
for j in range(n):
  #n-1 = len(weights)
  for i in range(n-1):
    s+= Data[j][0][i] * weights[i]
    m =round(s,2)
  c.append(m)
  s=0
  i=0
for i in range(n):
  p = Perceptron(i)
  y = Data[i][-1]
  print(f" actually target = {p} -------- our target = {y}")
  Data[i].append(p)
  Data[i].append(c[i])

print()


for i in Data:
  print(i)
print()
def Loss(Data):
  loss = 0
  for i in range(n):
    w_sum = Data[i][1]
    y = Data[i][-1]
    loss += round( -(w_sum*mt.log10(y) +(1-w_sum) * mt.log10(1 - y)),2)
  return loss/n
L = Loss(Data)
print(f"Loss = {L}") #distence between actually inputs and the Target



