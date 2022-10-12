#!/usr/bin/env python
# coding: utf-8

# In[5]:





import numpy as np

from scipy import sparse  


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
        return self.A @ self.u + dt*self.func(self.u)


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
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)



class Multistage(Timestepper):
    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stage = stages
        self.a = a
        self.b = b

    def _step(self, dt):
        #k = np.zeros((self.u.size, self.stages))
        k_mat = np.array([np.zeros(len(self.u)) for i in range(len(self.stages))])
        for i in range(self.stages):
            k_mat[:, i] = self.func(self.u +  (k_mat @ self.a[i, :])* dt)
        result = self.u + (k_mat @ self.b)*dt
        return result


class AdamsBashforth(Timestepper):
    
    def __init__(self, u, f, steps, dt):
            
        super().__init__(u, f)
        self.steps = steps
        self.firarr = []
        self.secarr = []
        self.coe = []
        for num in range(1, steps + 1):
            if len(self.coe) < num:
                coeffmat = np.zeros(num)
                for i in range(num):
                    polyno = np.array([1.0])
                    x1 = i / num
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


# In[ ]:




