import numpy as np, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from enum import Enum

# Generate data
numpy.random.seed(100) # skip when not debugging
classA = np.concatenate((np.random.randn(10 , 2) * 0.2+[1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5 ,0.5]))
classB = np.random.randn(20 , 2) * 0.2 + [0.0 , -0.5]
inputs = np.concatenate(( classA , classB ) )
t = np . concatenate((np.ones( classA.shape [0]) , -np.ones( classB . shape [0] )))
N = inputs.shape[ 0 ] # Number of rows ( samples )
permute = list( range(N) )
random.shuffle(permute)
inputs = inputs[permute, :]
t = t[permute]

def calculateP():
	return np.array([[t[i]*t[j]*kernel(inputs[i], inputs[j]) for j in range(N)] for i in range(N)])

P = calculateP()

# Linear kernel
def kernel(x ,y ):
    # linear
    return np.dot(x, y)





def objective(alpha):
    
    0.5*np.sum(np.sum(alpha*))
    return 0.5 * 

ret = minimize (objective, start, bounds=B, constraints=XC )
alpha = ret['x']



