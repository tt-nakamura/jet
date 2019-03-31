# reference:
#  F. Dias, A. R. Elcrat and L. N. Trefethen
#   "Ideal Jet Flow in Two Dimensions"
#   Journal of Fluid Mechanics 185 (1987) 275

import numpy as np
from scipy.special import roots_legendre, roots_jacobi
from scipy.optimize import root
import matplotlib.pyplot as plt

def plot_cmplx(z, *a, **k):
    plt.plot(np.real(z), np.imag(z), *a, **k)

class jet:
    def __init__(self, z, alpha, n_node=8, method='krylov'):
        """
        z = vertices as complex numbers on boundary
        alpha = interior angles between edges
        n_node = number of nodes for gaussian quadrature
        method = used for root finding
        """
        if len(z) < 3:
            print('there must be 3 or more vertices'); exit()
        if z.count(np.inf) != 1:
            print('there must be one Infinity'); exit()
        if np.isinf(z[0]) or np.isinf(z[-1]):
            print('z[0], and z[-1] must be finite'); exit()

        if np.isinf(z[1]):
            z.insert(1, z[0] + np.exp(-np.pi * alpha[0] * 1j))
            alpha[0] = 1

        n = len(z) - 1
        L = z.index(np.inf)

        beta = np.empty(n+1)
        gamma = np.angle(z[1] - z[0])/np.pi

        for k in range(1,n):
            if k==L or k+1==L:
                beta[k] = 1 - alpha[k+1-L]
            else:
                beta[k] = np.angle(z[k+1] - z[k])/np.pi - gamma
                beta[k] = (beta[k] + 1)%2 - 1
                if beta[k]-1 > 1.e-8: beta[k] = -1

            gamma += beta[k]

        beta[L] -= 1
        node = np.empty([n+1, n_node])
        weight = np.empty_like(node)

        for k in range(1,n):
            if k==L: continue
            (node[k], weight[k]) = roots_jacobi(n_node, 0, -beta[k])

        (node[0], weight[0]) = roots_jacobi(n_node, 0, 1)
        (node[n], weight[n]) = roots_legendre(n_node)

        self.prevertex = np.empty(n+1, dtype=np.complex)
        self.prevertex[0] = -1
        self.vertex = np.array(z, dtype=np.complex)
        self.angle = beta
        self.node = node
        self.weight = weight
        self.L = L
        self.map = np.vectorize(self.map)

        y = np.zeros(n-1)
        f = np.empty_like(y)

        def jfun(y):
            s = self.ystran(y)
            C = (z[1] - z[0])/self.squad(s[0], s[1], 0, 1)

            q = self.squad(s[0], 0.5j, 0) - self.squad(s[n], 0.5j, n)
            q = z[n] - z[0] - C*q
            f[L-2] = np.real(q)
            f[L-1] = np.imag(q)

            for k in range(1,n):
                if k==L or k+1==L: continue
                q = self.squad(s[k], s[k+1], k, k+1)
                f[k-1] = np.abs(z[k+1] - z[k]) - np.abs(C*q)

            self.C = C
            return f

        sol = root(jfun, y, method=method, options={'disp': True})
        self.ystran(sol.x)

        s = np.real(self.prevertex[1:-1])
        t = 1/(1 + s[L-1]**2)
        self.q = np.abs(self.C) * np.pi * t
        self.theta = np.angle(self.C)\
            - np.dot(beta[1:-1], np.pi/2 + 2 * np.arctan(s))
        t *= 2*s[L-1]
        self.w1 = self.q * (np.log((1+t)/(1-t))/np.pi + 1j)
        self.sigma_L = t

    def ystran(self,y):
        y = 1 + np.cumsum(np.exp(-np.cumsum(y)))
        self.prevertex[1] = 2/y[-1] - 1
        self.prevertex[2:] = 2*y/y[-1] - 1
        return self.prevertex

    def sprod(self,s,k=-1):
        n = len(self.vertex) - 1
        sk = self.prevertex[1:-1]
        sL = np.real(self.prevertex[self.L])
        s2 = s**2
        t = s - sk[:, np.newaxis]
        if k>0 and k<n: t[k-1] /= np.abs(t[k-1])
        t /= 1 - np.outer(sk,s)
        t = np.exp(-np.dot(self.angle[1:-1], np.log(t)))
        t /= (1+s2)*(s-sL)*(1-s*sL)
        if k==0: return t*(1-s)
        elif k==n: return -t*(1+s)
        else: return t*(1-s2)

    def dists(self,s,k):
        d = np.abs(s - self.prevertex)
        if k>=0: d[k] = np.inf
        return min(np.min(d), np.abs(s-1j))

    def sqsum(self,sa,sb,k):
        if sa==sb: return 0
        h = (sb-sa)/2
        if k==0 or k == len(self.vertex) - 1:
            t = self.sprod((sa+sb)/2 + h*self.node[0], k)
            t = h**2 * np.dot(self.weight[0], t)
        else:
            t = self.sprod((sa+sb)/2 + h*self.node[k], k)
            t = h * np.dot(self.weight[k], t)
            if k>0: t /= np.abs(h)**self.angle[k]
        return t

    def squad1(self,sa,sb,ka):
        if sa==sb: return 0
        q=0
        for _ in range(100):
            R = min(1, 0.5*self.dists(sa,ka)/np.abs(sb-sa))
            saa = sa + R*(sb-sa)
            q += self.sqsum(sa,saa,ka)
            if R==1: return q
            sa = saa
            ka = -1

        print('squad1 failed'); exit()

    def squad(self, sa, sb, ka=-1, kb=-1):
        sm = (sa+sb)/2
        return self.squad1(sa,sm,ka) - self.squad1(sb,sm,kb)

    def map(self, s, k=-1):
        sk = self.prevertex
        if k<0:
            d = np.abs(s - sk)
            d[self.L] = np.inf
            k = np.argmin(d)
        return self.vertex[k] + self.C * self.squad(sk[k], s, k)

    def plot(self, phi, n_stream, *arg, **kwarg):
        """
        phi = np.array of potential values
        n_stream = number of stream lines
        arg = arguments passed to plt.plot
        kwarg = keyword arguments passed to plt.plot
        """
        N = 100; EPS = 1.e-3

        p = phi*self.q
        q = np.arange(1, n_stream)/n_stream*self.q
        (p1,p2) = (np.min(p), np.max(p))

        def map_ws(w):
            sigma = 1 + (1 - self.sigma_L)*np.expm1(np.pi*w/self.q)
            return sigma/(1 + np.sqrt(1 - sigma**2))

        # free-stream lines
        w = np.linspace(0, p2, N) + EPS*1j
        plot_cmplx(self.map(map_ws(w)), *arg, **kwarg)
        w = np.linspace(self.w1, p2 + self.q*1j, N) - EPS*1j
        plot_cmplx(self.map(map_ws(w)), *arg, **kwarg)

        # stream lines
        x,y = np.meshgrid(np.linspace(p1,p2,N), q)
        z = self.map(map_ws(x+y*1j))
        plot_cmplx(z.T, *arg, **kwarg)

        if len(p)<=2: return

        # equipotential lines
        x,y = np.meshgrid(p, np.linspace(EPS, self.q-EPS, N))
        z = self.map(map_ws(x.T + y.T*1j))
        plot_cmplx(z.T, *arg, **kwarg)


    def plot_boundary(self, *arg, **kwarg):
        R = 10

        z = self.vertex[:self.L]
        gamma = np.angle(z[1] - z[0])/np.pi
        gamma += np.sum(self.angle[1:self.L])
        z = np.append(z, z[-1] + R*np.exp(np.pi * gamma * 1j))
        plot_cmplx(z, *arg, **kwarg)

        z = self.vertex[self.L+1:]
        gamma += 1 + self.angle[self.L]
        z = np.insert(z, 0, z[0] - R*np.exp(np.pi * gamma * 1j))
        plot_cmplx(z, *arg, **kwarg)
