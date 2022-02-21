import numpy as np, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from enum import Enum

debugging = True

# Generate data
if debugging:
    np.random.seed(100) # 100

spread = 0.2 # 0.2
classA = np.concatenate((np.random.randn(10 , 2) * spread+[1.5, 0.5], np.random.randn(10, 2) * spread + [-1.5 ,0.5]))
classB = np.random.randn(20 , 2) * spread + [0.0 , -0.5]
inputs = np.concatenate(( classA , classB ) )
t = np.concatenate((np.ones( classA.shape [0]) , -np.ones( classB.shape [0] )))
N = inputs.shape[ 0 ] # Number of rows ( samples )
permute = list( range(N) )
random.shuffle(permute)
inputs = inputs[permute, :]
t = t[permute]

# Linear kernel
def kernel(x ,y ):
    # linear
    return np.dot(x, y)

#def kernel(x,y):
	#polynomial
#	return 	np.dot(x,y)**2

def calculateP():
	return np.array([[t[i]*t[j]*kernel(inputs[i], inputs[j]) for j in range(N)] for i in range(N)])

def zerofun(alpha):
    return np.dot(alpha, t)

P = calculateP()
print("P", P)

def objective(alpha):
    sum = 0
    for i in range(N):
        for j in range(N):
            sum += alpha[i]*alpha[j]*P[i,j]
    sum  = 0.5*sum - np.sum(alpha)
    return sum
    """ test = 0.5 * np.dot(np.dot(P, alpha), alpha) - np.sum(alpha)
    return test """

alpha = np.zeros(N) # initial guess of the alpha vector

C = None
B =[(0,C) for b in range(N)]
XC = {'type':'eq', 'fun':zerofun}

ret = minimize(objective, alpha, bounds=B, constraints=XC )
#print"ret", ret)
alpha = ret['x']
# TODO HELP - alpha is only zeros. Where is the error?


# Extract non zero values of alpha
s = []
for i in range(len(alpha)):
    if alpha[i] > 1e-5:
        s.append(i)
    


def calculateB(s):
    #print"s", s)
    #print"type", type(s))
    a = [ alpha[i] for i in s]
    #print"a", a)       
    sum_me = [ alpha[i]*t[i]*kernel(alpha[i], inputs[i]) for i in s]
    return np.sum(sum_me)-np.sum([t[i] for i in s])
"""     a = [ alpha[i] for i in s]
    x = [ inputs[i] for i in s]
    t_s = [ t[i] for i in s] """

b = calculateB(s)

def ind(s):
    #print"s:", s, "b:", b)
    a = [ alpha[i] for i in s]
    x = [ inputs[i] for i in s]
    t_s = [ t[i] for i in s]
    return np.sum(a * t_s * kernel(s, x)) - b

def indicator(p):
    """ print("alpha", [alpha[i] for i in s])
    print("kernel", kernel(p[0], p[1]))
    print("t_i", [t[i] for i in s]) """
    sum_me = [ alpha[i]*t[i]*kernel(p[0], p[1])-t[i] for i in s]
    return np.sum(sum_me)


#plt.hold(True)
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

plt.axis('equal') # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen
plt.savefig('svmplot.pdf') # Save a copy in a file
#plt.show() # Show the plot on the screen


xgrid=np.linspace(-5, 5)
ygrid=np.linspace(-4, 4)
grid=np.array([[indicator([x, y]) for x in xgrid ] for y in ygrid])
"""print("grid")
for l in grid:
    print(l) """
plt.contour(xgrid , ygrid , grid , (-1.0, 0.0, 1.0),colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

plt.show()