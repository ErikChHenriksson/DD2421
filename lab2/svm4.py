import numpy as np, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from enum import Enum

debugging = True

# Generate data
if debugging:
    np.random.seed(100) # 100

# spread = 0.2 # 0.2
""" classA = np.concatenate((np.random.randn(10 , 2) * spread+[1.5, 0.5], np.random.randn(10, 2) * spread + [-1.5 ,0.5]))
classB = np.random.randn(20 , 2) * spread + [0.0 , -0.5] """

spread = 3 # 0.2
classA = np.concatenate((np.random.randn(5 , 2) * spread+[1.5, 0.5], np.random.randn(5, 2) * spread + [-1.5 ,0.5]))
classB = np.random.randn(5 , 2) * spread + [0.0 , -0.5]

# NEW
#spread = 0.2 # 0.2
#classA = np.random.randn(2 , 2)*spread + [ 2.0 , 1.0]
#classB = np.random.randn(2 , 2)*spread + [ -2.0 , -1.0]

inputs = np.concatenate(( classA , classB ) )
t = np.concatenate((np.ones( classA.shape [0]) , -np.ones( classB.shape [0] )))
N = inputs.shape[ 0 ] # Number of rows ( samples )
permute = list( range(N) )
#random.shuffle(permute)
inputs = inputs[permute, :]
t = t[permute]

def kernel(x, y, type="poly"):
    x = np.transpose(x)
    if type == "linear":
        return np.dot(x, y)
    elif type == "poly":
        return (np.dot(x,y)+1)**2
    elif type == "poly3":
        return (np.dot(x,y)+1)**3
    elif type == "RBF":
        sigma=1
        return math.exp(-np.linalg.norm(x-y, 2)**2/(2*sigma**2))
    else :
        return 0


def calculateP():
	return np.array([[t[i]*t[j]*kernel(inputs[i], inputs[j]) for j in range(N)] for i in range(N)])

def zerofun(alpha):
    return np.dot(alpha, inputs)

P = calculateP()
print("t: ",t)
print("P", P)
print("inputs: ", inputs)

def objective(alpha):
    sum = 0
    for i in range(N):
        for j in range(N):
            sum += alpha[i]*alpha[j]*P[i,j]
    sum  = 0.5*sum - np.sum(alpha)
    return sum
    """ test = 0.5 * np.dot(np.dot(P, alpha), alpha) - np.sum(alpha)
    return test """

start = np.zeros(N) # initial guess of the alpha vector

C = None
B =[(0,C) for b in range(N)]
XC = {'type':'eq', 'fun':zerofun}

ret = minimize(objective, start, bounds=B, constraints=XC )
print("ret", ret)
alpha = ret['x']
#print("alpha: ", alpha)

# Extract non zero values of alpha
s = []
for i in range(len(alpha)):
    if alpha[i] > 1e-5:
        s.append(i)

#print("s: ",s)

#New
def calculateB(s, bound = None): # HAVE TO BE BETWEEN 0 (AND C(if slack is used)) (not negative)
    index_b = s[0]
    if bound != None:
        for i in s:
            if alpha[i] < bound:
                index_b = i
                break
    a = [ alpha[i] for i in s]
    sum_me = [ alpha[i]*t[i]*kernel(inputs[index_b], inputs[i]) for i in s]
    return np.sum(sum_me)-t[index_b] # New

b = calculateB(s)
#print("b: ",b)

def ind(p):
    #print"s:", s, "b:", b)
    sum_me = [ alpha[i]*t[i]*kernel(p, inputs[i]) for i in s]
    return np.sum(sum_me) - b

def indicator(p):
    a = [ alpha[i] for i in s]

    #print("alpha", [alpha[i] for i in s])
    #print("p0,p1",p[0],p[1])
    #print("kernel", kernel(p[0], p[1]))
    #print("t_i", [t[i] for i in s])

    sum_me = [ alpha[i]*t[i]*kernel(a, p) for i in s]
    #print("ind", np.sum(sum_me) - b)
    return np.sum(sum_me) - b


#plt.hold(True)
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

plt.axis('equal') # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen
plt.savefig('svmplot.pdf') # Save a copy in a file
#plt.show() # Show the plot on the screen


""" xgrid=np.linspace(-5, 5)
ygrid=np.linspace(-4, 4) """
xgrid=np.linspace(-10, 10)
ygrid=np.linspace(-10, 10)
grid=np.array([[ind([x, y]) for x in xgrid ] for y in ygrid])
#print("grid", grid)

plt.contour(xgrid , ygrid , grid , (-1.0, 0.0, 1.0),colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

blue = mpatches.Patch(color='blue', label='Class A')
red = mpatches.Patch(color='red', label='Class B')
black = mpatches.Patch(color='black', label='Decision Boundry')
plt.legend(handles=[blue, red, black])

plt.xlabel('X')
plt.ylabel('Y')

plt.show()
