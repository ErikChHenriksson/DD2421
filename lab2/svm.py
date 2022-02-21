import numpy as np, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from enum import Enum

debugging = True

# Generate data
if debugging:
    np.random.seed(100)

classA = np.concatenate((np.random.randn(10 , 2) * 0.2+[1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5 ,0.5]))
classB = np.random.randn(20 , 2) * 0.2 + [0.0 , -0.5]
inputs = np.concatenate(( classA , classB ) )
t = np . concatenate((np.ones( classA.shape [0]) , -np.ones( classB . shape [0] )))
N = inputs.shape[ 0 ] # Number of rows ( samples )
permute = list( range(N) )
random.shuffle(permute)
inputs = inputs[permute, :]
t = t[permute]

# Linear kernel
def kernel(x ,y ):
    # linear
    return np.dot(x, y)

def kernel_pol(x,y,p):
	#polynomial
	return 	

def calculateP():
	return np.array([[t[i]*t[j]*kernel(inputs[i], inputs[j]) for j in range(N)] for i in range(N)])

def zerofun(alpha):
    return np.dot(alpha, t)

P = calculateP()

def objective(alpha):
    test = 0.5*np.dot(np.dot(P, alpha), alpha) - np.sum(alpha)

    return test

alpha = np.zeros(N) # initial guess of the alpha vector

C = None
B =[(0,C) for b in range(N)]
XC = {'type':'eq', 'fun':zerofun}

ret = minimize(objective, alpha, bounds=B, constraints=XC )
print("ret", ret)
alpha = ret['x']
# TODO HELP - alpha is only zeros. Where is the error?


# Extract non zero values of alpha
print("alpha", alpha)
s = []
for i in range(len(alpha)):
    if alpha[i] > 1e-5:
        s.append(i)
    


# TODO HELP - What is the correct way to write this? What inputs? np.sum correct?
def calculateB(s):
    print("s", s)
    print("type", type(s))
    a = [ alpha[i] for i in s]
    sum_me = [ alpha[i]*t[i]*kernel(a, inputs[i])-t[i] for i in s]
    return np.sum(sum_me)
"""     a = [ alpha[i] for i in s]
    x = [ inputs[i] for i in s]
    t_s = [ t[i] for i in s] """

b = calculateB(s)

# TODO HELP - What is the correct way to write this? What inputs? np.sum correct?
def ind(s):
    print("s:", s, "b:", b)
    a = [ alpha[i] for i in s]
    x = [ inputs[i] for i in s]
    t_s = [ t[i] for i in s]
    return np.sum(a * t_s * kernel(s, x)) - b



plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

plt.axis('equal') # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen
plt.savefig('svmplot.pdf') # Save a copy in a file
plt.show() # Show the plot on the screen


xgrid=np.linspace(-5, 5)
ygrid=np.linspace(-4, 4)
grid=np.array([[ind([x, y]) for x in xgrid ] for y in ygrid])
plt.contour(xgrid , ygrid , grid , (-1.0, 0.0, 1.0),colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))