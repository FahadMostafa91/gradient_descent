# -*- coding: utf-8 -*-
# gradient descent without tuning, no epsilon in the gamma = float(np.matmul((x-xprev).T,(d-dprev)))/den**2
# gamma = float(np.matmul((x-xprev).T,(d-dprev)))/den**2 + epsilon; if we add epsilon then it will tune

import numpy as np

def Rosenbrock(x):
    a = 2
    b = 100
    y = b*(x[1]-x[0]**2)**2+(x[0]-a)**2
    
    return y

def gradf(f,x,n,epsilon):
    y = np.zeros((n,1))
    for i in range (0,n):
        xperturbed = np.copy(x)
        xperturbed[i] = x[i] + epsilon
        y1 = f(xperturbed)
        xperturbed[i] = x[i] - epsilon
        y2 = f(xperturbed)
        y[i] = (y1-y2)/(2*epsilon)
    return y
    
def Gradient_Descent_BB(f,n):
    x = np.zeros((n,1))
    x[0] = float(1)
    x[1] = float(2)
    dprev = np.zeros((n,1))
    xprev = np.zeros((n,1))    
    tol = float(1e-9)
    epsilon = float(1e-8)
    
    kmax = int(1000)
    k = int(1)
    
    d = gradf(f,x,n,epsilon)
    f_x = f(x)
    s = np.linalg.norm(d)
    
    while (s > tol*(1+ np.absolute(f_x)) and k < kmax):
        if (k == 1):
            gamma = float(np.matmul(x.T,d))/s**2 + epsilon;
        else:
            den = np.linalg.norm(d-dprev)
            gamma = float(np.matmul((x-xprev).T,(d-dprev)))/den**2
        dprev = d
        xprev = x
        x = x - gamma*d
        f_x = f(x)
        d = gradf(f,x,n,epsilon)
        k = k+1
        s = np.linalg.norm(d)
    
    f_min = f_x;

    return x, f_min;
        
def main(): 
    n = 2
    x , f_min = Gradient_Descent_BB(Rosenbrock, n)
    print(x) 
    
    
    
if __name__ == "__main__":  ## This command executes the main function
    main()      
