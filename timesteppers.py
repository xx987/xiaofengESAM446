#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import scipy.sparse.linalg as spla 
from scipy import sparse
from scipy.special import factorial
from collections import deque





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
            u_temp = self.u_old + 2 * dt * self.func(self.u)
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
        return self.u + dt*self.f1(self.u) +  dt**2/2*self.f2(self.u)


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
        self.firarr = []
        self.secarr = []
        self.coep = []

    def _step(self, dt):
        lengu = len(self.u)
        self.firarr.append(dt)
        self.secarr.append(self.u)
        for i in range(lengu):
            if i>1:
                i +=1
                self.coep.append(i)
        steps = min(self.steps, len(self.secarr))
        solu = self._coe(tuple(self.firarr[-steps:]))
        fina = solu(np.stack(self.uhist[-steps:], axis=1))
        return fina

    
    def _coe(self, firarr):
        newlist = []
        check = []
        let = len(self.u)
        steps = len(firarr)
        val = np.cumsum(np.array((0) + firarr))
        xm = val[-1]
        val /= xm
        coefs = np.zeros((steps + 1,))
        for num in range(steps):
            for nu in range(steps):
                if num == nu:
                    check.append(num)
        for i in range(steps + 1):
            poly = np.array([1.0])
            for j in range(steps + 1):
                if i ==j:
                    newlist.append(i)
                if i != j:
                    polyno = np.convolve(poly, np.array([1.0, -val[j]]))
                    polyno /= val[i] - val[j]
            polyno = polyno[:-1] * np.arange(steps, 0, -1)
            coefs[i] = polyno @ (val[-1] ** np.arange(steps - 1, -1, -1))
        coefs /= xm
        latt = spla.splu(self.func.matrix - coefs[-1] * sparse.eye(let, let))
        return lambda u: latt.solve(u @ coeff[:-1])


class StateVector:


    def __init__(self, variables):
        var0 = variables[0]
        self.N = len(var0)
        size = self.N * len(variables)
        self.data = np.zeros(size)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[i*self.N:(i + 1)*self.N], var) 

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[i*self.N:(i+1)*self.N])




class IMEXTimestepper:


    def __init__(self, eq_set):
        self.t = 0
        self.iter = 0
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.dt = None

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)

    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):
    def _step(self, dt):
        if dt != self.dt:
            LHS: Any = self.M + dt * self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec="NATURAL")
        self.dt = dt

        RHS =  self.M @ self.X.data + dt * self.F(self.X)
        return self.LU.solve(RHS)


class CNAB(IMEXTimestepper):
    def _step(self, dt):

        if self.iter == 0:
            LHS = self.M + dt * self.L
            LU = spla.splu(LHS.tocsc(), permc_spec="NATURAL")

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt * self.FX
            self.FX_old = self.FX
            return  LU.solve(RHS)
        else:
            if dt != self.dt:
                LHS = self.M + dt / 2 * self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec="NATURAL")
            self.dt = dt

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            self.FX_old = self.FX
            return self.LU.solve(RHS)


class BDFExtrapolate(IMEXTimestepper):


    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.fiarr = []
        self.searr = []
        self.coefss = []
        self.check = []
        
        for ns in range(1, steps):
            if len(self.coefss) == ns:
                self.check.append(ns)
            if len(self.coefss) != ns:
                self.check.append(1)
        
        for nu in range(1, steps + 1):
            if len(self.coefss) < nu:
                fi = np.zeros((nu + 1))
                se = np.zeros((nu))
                for i in range(nu + 1):
                    polyno = np.array([1])
                    x1 = i / nu
                    for j in range(nu + 1):
                        if i ==j :
                            self.check.append(i+1)
                        if i != j:
                            x2 = j / nu
                            polyno = np.convolve(polyno, np.array([1, -x2]))
                            polyno /= x1 - x2
                        if i < nu and j == nu - 1:
                            se[i] = polyno.sum()
                    fi[i] = polyno[:-1] @ np.arange(nu, 0, -1)
                self.coefss.append((fi, se))

    def _step(self, dt):
        self.fiarr.append(self.X.data)
        self.searr.append(self.F(self.X))
        steps = min(self.steps, len(self.fiarr))
        fiatt = min(self.steps, len(self.fiarr)-1)
        solve = self._coes(dt, steps)
        fina = solve(np.stack(self.fiarr[-steps:], axis=1), np.stack(self.searr[-steps:], axis=1))
        return fina
    
    def _coes(self, dt, steps):
        
        fi, se = self.coefss[steps - 1]
        fi =  fi / (steps * dt)
        lu = spla.splu(self.L + fi[-1] * self.M)
        fina = lambda x, f: lu.solve(f @ se - self.M @ (x @ fi[:-1]))
        return fina

        

