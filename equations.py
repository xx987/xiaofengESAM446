#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy import sparse
from timesteppers import StateVector, CrankNicolson, RK22
import finite


class ReactionDiffusion2D:
    def __init__(self, c, D, dx2, dy2):
        self.t = 0
        self.iter = 0
        M, N = c.shape

        self.X = StateVector([c])
        self.F = lambda X: X.data * (1 - X.data)
        self.tstep = RK22(self)

        self.M = sparse.eye(M)
        self.L = -D * sparse.csc_array(dx2.matrix)
        self.xstep = CrankNicolson(self, 0)

        self.M = sparse.eye(N)
        self.L = -D * sparse.csc_array(dy2.matrix)
        self.ystep = CrankNicolson(self, 1)

    def step(self, dt):
        
        self.xstep.step(dt / 2)
        self.ystep.step(dt / 2)
        self.tstep.step(dt)
        self.ystep.step(dt / 2)
        self.xstep.step(dt / 2)
        self.t += dt
        self.iter += 1


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


        if isinstance(domain.grids[0], UniformPeriodicGrid):
            dx = DifferenceUniformGrid(1, evfun(spatial_order), domain.grids[0], 0)
            dsx = DifferenceUniformGrid(2, evfun(spatial_order), domain.grids[0], 0)
        else:
            dx = DifferenceNonUniformGrid(1, evfun(spatial_order), domain.grids[0], 0)
            dsx = DifferenceNonUniformGrid(2, odfun(spatial_order), domain.grids[0], 0)
            
        if checknumn(2) == evfun(2):
            taa=[3,4,5]
        else:
            taa=[5,5,5]
            

        if isinstance(domain.grids[1], UniformPeriodicGrid):
            dy = DifferenceUniformGrid(1, evfun(spatial_order), domain.grids[1], 1)
            dsy = DifferenceUniformGrid(2, evfun(spatial_order), domain.grids[1], 1)
        else:
            dy = DifferenceNonUniformGrid(1, evfun(spatial_order), domain.grids[1], 1)
            dsy = DifferenceNonUniformGrid(2, odfun(spatial_order), domain.grids[1], 1)

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
                [nu * d2
                 dsx.matrix, sparse.csr_matrix((M, M))],
                [sparse.csr_matrix((M, M)), nu * dsx.matrix],
            ]
        )
        self.xstep = CrankNicolson(self, 0)

        self.X = StateVector([u, v], 1)
        self.M = sparse.eye(2 * N)
        self.L = sparse.bmat(
            [
                [nu * d2y.matrix, sparse.csr_matrix((N, N))],
                [sparse.csr_matrix((N, N)), nu * d2y.matrix],
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
    def __init__(self, u, p, du, dp, rho0, gamma_p0):
        (N,) = u.shape
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))
        self.X = StateVector([u, p])
        self.M = sparse.bmat([[rho0 * sparse.csc_array(I), Z], [Z, I]])
        self.L = sparse.bmat([[Z, d.matrix], [gammap0 * sparse.csc_array(d.matrix), Z]])
        self.F = lambda X: np.zeros(X.data.shape)


class SoundWave:
    
    def __init__(self, u, p, d, rho0, gammap0):
        leng = len(u)
        Z = sparse.csr_matrix((leng, leng))
        I = sparse.eye(leng, leng)  
        self.X = StateVector([u, p])
        self.M = sparse.bmat([[rho0 * I, Z], [Z, I]])
        self.L = sparse.bmat([[Z, d.matrix], [gammap0 * d.matrix, Z]])
        self.F = lambda X: np.zeros(X.data.shape)


class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        
        
        leng = len(c)
        self.L = -D * d2.matrix
        self.X = StateVector([c])
        self.M = sparse.eye(leng, leng)
        self.F = lambda X: X.data * (c_target - X.data)

