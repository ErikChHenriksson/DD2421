import numpy as np, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from enum import Enum

debugging = True

# Generate data
if debugging:
    np.random.seed(100) # 100

# ------------------------------------------------------------------------
# GENERATING DATA
classA = np.concatenate((np.random.randn (10 , 2) * 0.2 + [ 1.5 , 0.5 ] ,np.random.randn(10 , 2)*0.2 + [ -1.5 , 0.5 ]))
classB = np.random.randn(20 , 2)*0.2 + [ 0.0 , -0.5]

inputs = np.concatenate((classA , classB))
targets = np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows ( samples )
permute = list(range(N))
random.shuffle( permute )
inputs = inputs[permute , : ]
targets = targets[ permute ]

print(targets)
# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
# Minimize will find the vector alpha that minimize the function objective
# with the bounds B and the constraints XC
def generate_data(start):
    XC={'type':'eq', 'fun':zerofun} # function in zerofun must equal zero as in eq(10)
    #B=[(0, C) for b in range(N)] # lower and upper bound
    B=[(0,None) for b in range(N)] # if only lower bound is wanted
    ret = minimize(objective , start , bounds=B, constraints=XC)
    alpha = ret['x']
    print(ret)
    print(ret['success'])
    print(alpha)
    return alpha
# ------------------------------------------------------------------------


# Return scalar value, eq(4)
def objective(alpha):
    objective_sum = 0.5 * np.dot(np.dot(P, alpha), alpha) - np.sum(alpha)
    return objective_sum # =alpha

# Return scalar value, eq (10)
def zerofun(alpha):
    return np.dot(alpha,targets)

def kernel(x,y): # Linear
    return np.dot(x,y)

#def kernel(x,y): # Polynomial
#    return (np.dot(np.transpose(x), y) + 1) ** N

def P_matrix(alpha):
    P = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            P[i][j] = targets[i] * targets[j] * kernel(alpha[i],alpha[j])
    return P

###### IMPORTANT
 # must use alpha larger then 0 and smaller then C
def calculate_b(s):
    a = [ alpha[i] for i in s]
    sum_me = [ alpha[i]*targets[i]*kernel(alpha[i], inputs[i]) for i in s]
    return np.sum(sum_me)-np.sum([targets[i] for i in s])

def ind(s):
    a = [ alpha[i] for i in s]
    x = [ inputs[i] for i in s]
    t_s = [ targets[i] for i in s]
    return np.sum(a * t_s * kernel(s, x)) - b

def ind2(p):
    """ print("alpha", [alpha[i] for i in s])
    print("kernel", kernel(p[0], p[1]))
    print("t_i", [t[i] for i in s]) """
    sum_me = [ alpha[i]*targets[i]*kernel(p[0], p[1])-targets[i] for i in s]
    return np.sum(sum_me)

# Generate the data from given code
alpha = np.zeros(N) # =start
P = P_matrix(alpha)
print(P)
alpha = generate_data(alpha)
print('\n')

# only keep the non-zero values in alpha

print(alpha)
print('\n')

# Extract non zero values of alpha
s = []
for i in range(len(alpha)):
    if alpha[i] > 1e-5:
        s.append(i)

print(s)
b = calculate_b(s) # b is a float


# ---------------------------------------------------------
# Plotting

plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

plt.axis('equal') # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen
plt.savefig('svmplot.pdf') # Save a copy in a file
#plt.show() # Show the plot on the screen


xgrid=np.linspace(-5, 5)
ygrid=np.linspace(-4, 4)
grid=np.array([[ind2([x, y]) for x in xgrid ] for y in ygrid])
"""print("grid")
for l in grid:
    print(l) """
plt.contour(xgrid , ygrid , grid , (-1.0, 0.0, 1.0),colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

plt.show()
