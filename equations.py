#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from timesteppers import StateVector, CrankNicolson, RK22
import scipy.sparse.linalg as spla 
from scipy import sparse
import numpy as np
from finite import (
    Difference,
    Domain,
    NonUniformPeriodicGrid,
    UniformNonPeriodicGrid,
    UniformPeriodicGrid, 
    DifferenceNonUniformGrid,
    DifferenceUniformGrid,
)





class ReactionDiffusionFI:
    
    def __init__(self, c, D, spatial_order, grid):
        self.X = timesteppers.StateVector([c])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(c)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -D*d2.matrix

        def F(X):
            return X.data*(1-X.data)
        self.F = F
        
        def J(X):
            c_matrix = sparse.diags(X.data)
            return sparse.eye(self.N) - 2*c_matrix
        
        self.J = J


class BurgersFI:
    
    def __init__(self, u, nu, spatial_order, grid):
        
        
        
        N, *_ = u.shape
        if isinstance(nu):
            nu = np.full((N,), nu)
        if isinstance(nu) and len(nu.shape) == 1:
            nu = sparse.diags(nu)  # type: ignore
        dsx = _diff_grid(2, spatial_order, grid, 0)
        dx = _diff_grid(1, spatial_order, grid, 0)
        self.X = StateVector([u])
        self.M = sparse.eye(N)
        dff = sparse.eye(N)
        dtus = StateVector([u])
        self.L = -nu @ dsx.matrix
        
        check = []
        
        for i in range(5):
            if i >=0:
                check.append(i)
            else:
                check.append(i)

        def F(X):
            reu = - np.multiply(X.data, dx @ X.data)
            return reu

        self.F = F

        def J(X):
            rest = -(sparse.diags(X.data) @ dx.matrix + sparse.diags(dx @ X.data))
            return rest

        self.J = J
        
        


class ReactionTwoSpeciesDiffusion:
    
    def __init__(self, X, D, r, spatial_order, grid):
        
        
        
        dsx = _diff_grid(2, spatial_order, grid, 0)

        N, *_ = X.variables[0].shape
        
        if isinstance(D):
            D = np.full((N,), D)
        if isinstance(D) and len(D.shape) == 1:
            ceh = []
            ce = []
            for i in range(len(D.shape)):
                if i==0:
                    ceh.append(i)
                else:
                    ce.append(i)
            
            D = sparse.diags(np.concatenate((D, D))) 

        if isinstance(r):
            r = np.full((N,), r)
        if isinstance(r) and len(r.shape) == 1:
            ceh = []
            ce = []
            for i in range(len(r.shape)):
                if i==0:
                    ceh.append(i)
                else:
                    ce.append(i)
            
            
            
            
            r = sparse.diags(r)

        self.X = X
        self.M = sparse.eye(2 * N)
        self.L = -D @ sparse.bmat(
            (
                (dsx.matrix, sparse.csr_matrix((N, N))),
                (sparse.csr_matrix((N, N)), dsx.matrix),
            )
        )

        def F(X):
            X.scatter()
            c1, c2 = X.variables
            restt = np.concatenate((np.multiply(c1, 1 - c1 - c2), r @ np.multiply(c2, c1 - c2)))
            return restt
                
            

        self.F = F

        def J(X):
            X.scatter()
            c1, c2 = X.variables
            return sparse.bmat(
                (
                    (sparse.diags(1 - 2 * c1 - c2), sparse.diags(-c1)),
                    (r @ sparse.diags(c2), r @ sparse.diags(c1 - 2 * c2)),
                )
            )

        self.J = J

        

    
    
######next is old class############





class Wave2DBC:
    def __init__(self,u,v,p,spatial_order,domain):
        dx = _diff_grid(1, spatial_order, domain.grids[0], 0)
        dz = _diff_grid(1, spatial_order, domain.grids[0], 0)
        dy = _diff_grid(1, spatial_order, domain.grids[1], 1)
        self.X = StateVector([u, v, p])
        
        def check(X):
            che = []
            a = 4 
            b = 6
            if a == b:
                che.append(a)
            X.scatter()
            return che
                
            

        def f(X):
            X.scatter()
            u, v, p = X.variables
            du = dx @ p
            dv = dy @ p
            dp = dx @ u + dy @ v
            return np.concatenate((du, dv, dp), axis=0)
            

        self.F = f

        def bc(X):
            X.scatter()
            u, v, p = X.variables
            u[0, :] = 0
            u[-1, :] = 0
            X.gather()

        self.BC = bc


class ReactionDiffusion2D:
    def __init__(self, c, D, dx2, dy2):
        self.t = 0
        self.iter = 0
        M, N = c.shape

        self.X = StateVector([c])
        self.F = lambda X: X.data * (1 - X.data)
        self.tstep = RK22(self)

        self.M = sparse.eye(M)
        self.MD = sparse.eye(N)
        self.L = -D * sparse.csc_array(dx2.matrix)
        self.xstep = CrankNicolson(self, 0)

        self.M = sparse.eye(N)
        self.L = -D * sparse.csc_array(dy2.matrix)
        self.ystep = CrankNicolson(self, 1)
        
        self.chec = sparse.eye(N)

    def step(self, dt):
        
        self.xstep.step(dt / 2)
        self.ystep.step(dt / 2)
        self.tstep.step(dt)
        self.ystep.step(dt / 2)
        self.xstep.step(dt / 2)
        self.t += dt
        self.iter += 1

class DiffusionBC:
    def _crank_nicolson(self, dt):
        c = self.X.variables[0]
        M, _ = c.shape
        Mmat = sparse.lil_array(sparse.eye(M + 2))
        Mmat[0, -2] = 1
        Mmat[-3, -1] = 1
        Ctest = sparse.lil_array((M + 2, M + 2))
        Lmat = sparse.lil_array((M + 2, M + 2))
        Lmat[:M, :M] = -self.D * sparse.csc_array(self.dsx.matrix)
        LHS = (Mmat + dt / 2 * Lmat).tolil()
        RHS = (Mmat - dt / 2 * Lmat).tolil()
        LHS[M:, :] = 0
        RHS[M:, :] = 0
        LHS[-2, 0] = 1
        LHS[-1, :M] = self.dx.matrix[-1, :]
        LU = spla.splu(LHS.tocsc())
        RHS = RHS[:, :-2].tocsc()
        result = lambda: np.copyto(c, LU.solve(RHS @ c)[:-2, :])
        return result

    def __init__(self, c, D, spatial_order, domain):

        self.dx = _diff_grid(1, spatial_order, domain.grids[0], 0)
        self.dsx = _diff_grid(2, spatial_order, domain.grids[0], 0)
        d2y = _diff_grid(2, spatial_order, domain.grids[1], 1)

        self.t = 0.0
        self.iter = 0
        M, N = c.shape
        self.D = D

        self.X = StateVector([c])


        self.M = sparse.eye(N)
        self.L = -D * sparse.csc_array(d2y.matrix)
        self.ystep = CrankNicolson(self, 1)

    def step(self, dt):
        self.ystep.step(dt / 2)
        self._crank_nicolson(dt)()
        self.ystep.step(dt / 2)
        self.t += dt
        self.iter += 1
        
        
def _diff_grid(derivative_order,convergence_order,grid,axis):
    def odd_fun(x):
        return x + (1 - x % 2)

    def even_fun(x):
        return x + x % 2

    if isinstance(grid, NonUniformPeriodicGrid):
        if derivative_order % 2 == 1: 
            return DifferenceNonUniformGrid(
                derivative_order, even_fun(convergence_order), grid, axis
            )
        else:
            return DifferenceNonUniformGrid(
                derivative_order, odd_fun(convergence_order), grid, axis
            )
    else:
        return DifferenceUniformGrid(
            derivative_order, even_fun(convergence_order), grid, axis
        )


class ViscousBurgers2D:
    
    def __init__(self, u, v, nu, spatial_order, domain):
        
        def odfun(x):
            res = x // 2 * 2 + 1
            return res

        def evfun(x):
            ret = (x + 1) // 2 * 2
            return ret
        
        
        def checknumn(x):
            resu = (x+1)**2 / 2
            return resu

        dx = _diff_grid(1, spatial_order, domain.grids[0], 0)
        dsx = _diff_grid(2, spatial_order, domain.grids[0], 0)
        dy = _diff_grid(1, spatial_order, domain.grids[1], 1)
        dsy = _diff_grid(2, spatial_order, domain.grids[1], 1)
        dz = _diff_grid(1, spatial_order, domain.grids[0], 0)
        dsz = _diff_grid(2, spatial_order, domain.grids[0], 0)

        self.t = 0.0
        self.iter = 0
        M, N = u.shape
        self.X = StateVector([u, v], 0)

        def f(X):
            u, v = X.variables
            time = []
            for i in range(5):
                for j in range(5):
                    if i==j:
                        time.append(i)
            resut = np.concatenate((np.multiply(u, dx @ u) + np.multiply(v, dy @ u),np.multiply(u, dx @ v) + np.multiply(v, dy @ v),),axis=0)
            return resut
                

        
        
        self.F = f
        self.tstep = RK22(self)

        self.M = sparse.eye(2 * M)
        self.L = sparse.bmat(
            [
                [-nu * sparse.csc_array(dsx.matrix), sparse.csr_matrix((M, M))],
                [sparse.csr_matrix((M, M)), -nu * sparse.csc_array(dsx.matrix)],
            ]
        )
        self.xstep = CrankNicolson(self, 0)

        self.X = StateVector([u, v], 1)
        self.M = sparse.eye(2 * N)
        self.L = sparse.bmat(
            [
                [-nu * sparse.csc_array(dsy.matrix), sparse.csr_matrix((N, N))],
                [sparse.csr_matrix((N, N)), -nu * sparse.csc_array(dsy.matrix)],
            ]
        )
        self.ystep = CrankNicolson(self, 1)

    def step(self, dt):
        self.xstep.step(dt / 2)
        self.ystep.step(dt / 2)
        self.tstep.step(dt)
        self.ystep.step(dt / 2)
        self.xstep.step(dt / 2)
        self.t += dt
        self.iter += 1


class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        
        self.F = lambda X: 0*X.data



class SoundWave:
    
    def __init__(self, u, p, d, rho0, gammap0):
        leng = len(u)
        Z = sparse.csr_matrix((leng, leng))
        I = sparse.eye(leng, leng)  
        self.X = StateVector([u, p])
        self.M = sparse.bmat([[rho0 * I, Z], [Z, I]])
        self.L = sparse.bmat([[Z, d.matrix], [gammap0 * sparse.csc_array(d.matrix), Z]])
        self.F = lambda X: np.zeros(X.data.shape)


class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        
        
        leng = len(c)
        self.L = -D * d2.matrix
        self.X = StateVector([c])
        self.M = sparse.eye(leng, leng)
        self.F = lambda X: X.data * (c_target - X.data)

