# reference:
#  A. R. Elcrat and L. N. Trefethen
#   "Classical Free-Streamline Flow over a Polygonal Obstacle"
#   Journal of Computational and Applied Mathematics 14 (1986) 251

import numpy as np
from scipy.special import roots_legendre, roots_jacobi
from scipy.optimize import root
import matplotlib.pyplot as plt

class kirch:
    def __init__(self, z, n_node=8, method='krylov'):
        """
        z = vertices as complex numbers on boundary
        n_node = number of nodes for gaussian quadrature
        method = used for root finding
        """
        if len(z)<2:
            print('there must be 2 or more vertices'); exit()
        if len(z)==2:
            z.insert(1, (z[0]+z[1])/2)

        n = len(z) - 1

        beta = np.empty(n+1)
        node = np.empty([n+1, n_node])
        weight = np.empty_like(node)

        (node[n], weight[n]) = roots_legendre(n_node)
        (node[0], weight[0]) = (node[n], weight[n])
        gamma = np.angle(z[1] - z[0])/np.pi

        for k in range(1,n):
            beta[k] = np.angle(z[k+1] - z[k])/np.pi - gamma
            beta[k] = (beta[k] + 1)%2 - 1
            if beta[k]-1 > 1.e-8: beta[k] = -1

            gamma += beta[k]
            (node[k], weight[k]) = roots_jacobi(n_node, 0, -beta[k])

        self.prevertex = np.empty(n+1, dtype=np.complex)
        self.prevertex[0] = -1
        self.vertex = np.array(z, dtype=np.complex)
        self.angle = beta
        self.node = node
        self.weight = weight
        self.reflect = False
        self.map = np.vectorize(self.map)

        y = np.zeros(n-1)
        f = np.empty_like(y)

        def kfun(y):
            x = self.yxtran(y)
            C = (z[1] - z[0])/self.xquad(x[0], x[1], 0, 1)

            for k in range(1,n):
                q = self.xquad(x[k], x[k+1], k, k+1)
                f[k-1] = np.abs(z[k+1] - z[k]) - np.abs(C*q)

            self.C = C
            return f

        sol = root(kfun, y, method=method, options={'disp': True})
        self.yxtran(sol.x)

    def yxtran(self,y):
        x = self.prevertex
        y = 1 + np.cumsum(np.exp(-np.cumsum(y)))
        x[1] = 2/y[-1] - 1
        x[2:] = 2*y/y[-1] - 1

        self.stag = -np.cos(
            np.angle(self.vertex[-1] - self.vertex[-2])
            - np.dot(self.angle[1:-1], np.arccos(-x[1:-1]))
        )

        return x

    def xprod(self,x,k=-1):
        xk = self.prevertex[1:-1]
        x2 = 1 - x**2
        t = x - xk[:, np.newaxis]
        if k>0: t[k-1] /= np.abs(t[k-1])
        t /= 1 - np.outer(xk,x) + np.sqrt(np.outer(1-xk**2, x2))
        t = np.exp(-np.dot(self.angle[1:-1], np.log(t)))
        t *= 1 - self.stag * x + np.sqrt(x2*(1 - self.stag**2))
        if self.reflect: return (x - self.stag)**2 / t
        else: return t

    def distx(self,x,k):
        d = np.abs(x - self.prevertex)
        if k>=0: d[k] = np.inf
        return np.min(d)

    def xqsum(self,xa,xb,k):
        if xa==xb: return 0
        h = (xb-xa)/2
        if k==0 or k == len(self.vertex) - 1:
            t = self.node[k] + 1
            t = self.xprod(xa + h * t**2/2) * t
        else:
            t = self.xprod((xa+xb)/2 + h*self.node[k], k)
            if k>0: h /= np.abs(h)**self.angle[k]
        return h * np.dot(self.weight[k], t)

    def xquad1(self,xa,xb,ka):
        if xa==xb: return 0
        q=0
        for _ in range(100):
            R = min(1, self.distx(xa,ka)/np.abs(xb-xa))
            xaa = xa + R*(xb-xa)
            q += self.xqsum(xa,xaa,ka)
            if R==1: return q
            xa = xaa
            ka = -1

        print('xquad1 failed'); exit()

    def xquad(self, xa, xb, ka=-1, kb=-1):
        xm = (xa+xb)/2
        return self.xquad1(xa,xm,ka) - self.xquad1(xb,xm,kb)

    def map(self, x, k=-1):
        xk = self.prevertex
        if k<0: k = np.argmin(np.abs(x - xk))
        return self.vertex[k] + self.C * self.xquad(xk[k], x, k)

    def force(self):
        self.reflect = True
        t = self.map(-1j, 0)
        t -= self.C * self.xquad(1, -1j, len(self.vertex)-1)
        self.reflect = False
        return (t - self.vertex[-1]) * 1j

    def plot(self, phi, psi, *arg, **kwarg):
        """
        phi = np.array of potential values
        psi = np.array of stream function values
        arg = arguments passed to plt.plot
        kwarg = keyword arguments passed to plt.plot
        """
        N = 100; EPS = 1.e-4

        p = phi[np.abs(phi)>EPS]/np.abs(self.C)
        q = psi[np.abs(psi)>EPS]/np.abs(self.C)

        (p1,p2) = (np.min(p), np.max(p))
        (q1,q2) = (np.min(q), np.max(q))

        def map_wx(w):
            x = np.sqrt(2*w)
            x[np.imag(x) < 0] *= -1
            return x + self.stag

        def plot_cmplx(z, *a, **k):
            plt.plot(np.real(z), np.imag(z), *a, **k)

        # free-stream lines
        for i in [-1,1]:
            x = np.linspace(i, i*np.sqrt(2*p2) + self.stag, N)
            plot_cmplx(self.map(x + EPS*1j), *arg, **kwarg)

        # stream line to stagnation point
        z = self.map(map_wx(np.linspace(0,p1+0j,N)))
        plot_cmplx(z, *arg, **kwarg)

        # internal stream lines
        x,y = np.meshgrid(np.linspace(p1,p2,N), q)
        plot_cmplx(self.map(map_wx(x+y*1j)).T, *arg, **kwarg)

        if len(p)<=2: return

        # upstream equipotential lines
        x,y = np.meshgrid(p[p<0], np.linspace(q1,q2,N))
        plot_cmplx(self.map(map_wx(x.T+y.T*1j)).T, *arg, **kwarg)

        # equipotential line to stagnation point
        for w in [q1*1j, q2*1j]:
            z = self.map(map_wx(np.linspace(0,w,N)))
            plot_cmplx(z, *arg, **kwarg)

        # downstream equipotential lines
        for q in [q1,q2]:
            x,y = np.meshgrid(p[p>0], q * np.linspace(EPS,1,N))
            plot_cmplx(self.map(map_wx(x.T+y.T*1j)).T, *arg, **kwarg)
