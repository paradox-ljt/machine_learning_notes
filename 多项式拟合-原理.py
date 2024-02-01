import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def coefficient(M, N, x, y,lamda=0):    # M is the order number
    print("  M=%d  ,  N=%d  "%(M,N))
    order = np.arange(M+1)    #np.arange->list
    order = order[:, np.newaxis] #Increased dimension becomes a column matrix
    e = np.tile(order, [1,N])  #Copy N times along the y-axis
    XT = np.power(x, e)  #Build a matrix that looks like a Van der Monde determinant
    X = np.transpose(XT)  
    m = np.matmul(XT, X) + lamda*np.identity(M+1)  #XT * X
    n = np.matmul(XT, y) #XT * y
    w = np.linalg.solve(m,n) #mW = n -> W = (XT * X)^-1 * XT * y
    print("W:")   #Coefficient matrix
    print(w)
    return w

def fit(M,w):
    order = np.arange(M+1)    #np.arange->list
    order = order[:, np.newaxis] #Increased dimension becomes a column matrix
    e2 = np.tile(order, [1,x.shape[0]])
    XT2 = np.power(x, e2)
    p = np.matmul(w, XT2)
    print("p:")
    print(p)
    return p

# prepare data
excel = pd.read_excel(r"C:/Users/华为/Desktop/程序/多项式回归-test.xlsx")
x = np.array(excel['a']).astype(np.float)
y = np.array(excel['b']).astype(np.float)
x = np.array(x)
y = np.array(y)
N = len(x)   # Number of data

#Get the coefficient matrix
w = coefficient(74, N, x, y)  #10 order fitting
p = fit(74,w)

#drawing picture
plt.figure(1, figsize=(8,5))
plt.plot(x, y, 'lightcoral', x, p, 'aquamarine',linewidth=3) 
#plt.scatter(x, y,  marker='o', edgecolors='b', s=100, linewidth=3)
plt.text(0.8, 0.9,'M = 10', style = 'italic')
plt.title('Figure 1 : M = 10, N = 10')
plt.savefig('1.png', dpi=400)
plt.show()