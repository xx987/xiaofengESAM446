#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np

from scipy import sparse  
from scipy.special import factorial  


class UniformPeriodicGrid:
    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:
    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class DifferenceUniformGrid:
    def __init__(self, derivative_order, convergence_order, grid, stencil_type: str = "centered"):
        
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.matrix = DifferenceNonUniformGrid(derivative_order, convergence_order, NonUniformPeriodicGrid(grid.values, grid.length),stencil_type).matrix

    def __matmul__(self, other):
        return self.matrix @ other


class DifferenceNonUniformGrid:
    def __init__(self, derivative_order, convergence_order, grid, stencil_type: str = "centered"):
        
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        
        
        npoints = convergence_order + derivative_order  + 1
        
        
        orimatrix = sparse.dok_matrix((grid.N, grid.N))
        
        ###############for special case detection
        
        if grid.N != grid.N:
            orimatrix = sppaese.dok.matrix(((grid.N)-1, (grid.N)-1))
            for i in range(len(grid.values)):            
                xt = grid.values[i]           
                for i in range(len(orimatrix)):
                    if self.derivative_order ==2:
                        orimatrix[i][i] = -1
                        orimatrix[i][i+1]= 1
                    if self.derivative_order ==4:
                        orimatrix[i][i] = 1
                        orimatrix[i][i+1]= -2
                        orimatrix[i][i+2]= 3
            matrix == orimatrix
                                                        
        ##################
        for i in range(len(grid.values)):            
            xt = grid.values[i]           
            allrange = range(i - npoints // 2, i + npoints // 2 + 1)
            for ele in allrange:  
                func = np.zeros((derivative_order + 1))
                func[0] = 1
                xj = (ele // grid.N) * grid.length + grid.values[ele % grid.N] - xt
                for j in allrange:  
                    if j != ele:
                        xk = (j // grid.N) * grid.length + grid.values[j % grid.N] - xt
                        func[1:] = (xk * func[1:] - func[:-1]) / (xk - xj)
                        func[0] *= xk / (xk - xj)
                orimatrix[i, ele % grid.N] += func[-1]
                
        
        matrix = orimatrix.tocsr()
        matrix *= factorial(derivative_order)
        self.matrix = matrix

    def __matmul__(self, other):
        return self.matrix @ other


