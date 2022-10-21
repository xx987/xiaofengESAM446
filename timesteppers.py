#!/usr/bin/env python
# coding: utf-8

# In[1]:






import numpy as np
import scipy.sparse.linalg as spla  
from scipy import sparse

from finite import DifferenceNonUniformGrid, DifferenceUniformGrid

#T = TypeVar("T")


class Timestepper:
    


    def __init__(self, u, f):
        self.t = 0
        self.iter = 0
        self.u = u
        self.func = f
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ForwardEuler(Timestepper):
    def _step(self, dt):
        return self.u + dt * self.func(self.u)


class LaxFriedrichs(Timestepper):
    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1 / 2, 1 / 2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1 / 2
        A[-1, 0] = 1 / 2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt * self.func(self.u)


class Leapfrog(Timestepper):
    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt * self.func(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.func(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):
    
    def __init__(self, u, func1, func2):
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = func1
        self.f2 = func2

    def _step(self, dt):
        return self.u + dt * self.f1(self.u)+ dt**2 / 2 * self.f2(self.u)
        


class Multistage(Timestepper):
    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b

    def _step(self, dt):
        k_mat = np.zeros((len(self.u), self.stages))
        umat = []
        for j in range(len(self.u)):
            if self.u[0]> self.u[j-1]:
                umat.append(self.u[j])
            else:
                umat.append(self.u[j])
            
        for i in range(self.stages):
            k_mat[:, i] = self.func(self.u +  (k_mat @ self.a[i, :])* dt)
            newmat = (k_mat @ self.a[i, :])* dt
            umat.append(newmat[0])
        result = self.u + (k_mat @ self.b)*dt
        return result


class AdamsBashforth(Timestepper):
    
    def __init__(self, u, f, steps, dt):
            
        super().__init__(u, f)
        self.steps = steps
        self.firarr = []
        self.secarr = []
        self.coe = []
        for nut in range(steps):
            fcoef = []
            prod = 1 / (nut+1)
            fcoef.append(prod)
            if prod >1:
                newcoef = np.zeros(nut)
                newcoef[0] = 1

        for num in range(1, steps + 1):
            if len(self.coe) < num:
                coeffmat = np.zeros(num)
                coestr = np.zeros(num)
                for i in range(num):
                    polyno = np.array([1.0])
                    x1 = i / num
                    if x1 == 0:
                        xre = x1
                        xrs = x1+1
                    for j in range(num):
                        if i != j:
                            x2 = j / num
                            polyno = np.convolve(polyno, np.array([1.0, -x2]))
                            polyno /= x1 - x2
                    polyno /= np.arange(num, 0, -1)
                    coeffmat[i] = polyno.sum()
                self.coe.append(coeffmat)

    def _step(self, dt):
        self.firarr.append(self.u)
        self.secarr.append(self.func(self.u))
        steps = min(self.steps, len(self.firarr))
        finres = self.firarr[-steps] + (steps* dt* (np.stack(self.secarr[-steps:], axis=1) @ self.coe[steps - 1]))
        return finres


class BackwardEuler(Timestepper):
    def __init__(self, u, L):
        super().__init__(u, L)
        N = len(u)
        self.I = sparse.eye(N, N)

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt * self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec="NATURAL")
        self.dt = dt
        return self.LU.solve(self.u)


class CrankNicolson(Timestepper):
    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        N = len(u)
        self.I = sparse.eye(N, N)

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt / 2 * self.func.matrix
            self.RHS = self.I + dt / 2 * self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec="NATURAL")
        self.dt = dt
        return self.LU.solve(self.RHS @ self.u)


class BackwardDifferentiationFormula(Timestepper):

    def __init__(self, u, L_op, steps):
        super().__init__(u, L_op)
        self.steps = steps
        #self.thist = []
        #self.uhist = []
        self.firarr = []
        self.secarr = []
        self.coe = []

    def _step(self, dt):
        (N,) = self.u.shape
        self.firarr.append(dt)
        self.secarr.append(self.u)
        steps = min(self.steps, len(self.firarr))
        coeff = BackwardDifferentiationFormula._coeff(tuple(self.firarr[-steps:]))
        return spla.spsolve(self.func.matrix - coeff[-1] * sparse.eye(N, N),np.stack(self.secarr[-steps:], axis=1)@ coeff[:-1])


    def _coeff(firarr):
        steps = len(firarr)
        x = np.cumsum(np.array((0,) + thist))
        coeff = np.zeros((steps + 1,))
        for i in range(steps + 1):
            poly = np.array([1.0])
            for j in range(steps + 1):
                if i != j:
                    poly = np.convolve(poly, np.array([1.0, -x[j]]))
                    poly /= x[i] - x[j]
            poly = poly[:-1] * np.arange(steps, 0, -1)
            coeff[i] = poly @ (x[-1] ** np.arange(steps - 1, -1, -1))
        return coeff


# In[ ]:




