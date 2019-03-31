import numpy as np
from jet import jet
import matplotlib.pyplot as plt

plt.figure(figsize=(6.4,4.8))

n_stream = 10
phi = np.linspace(-1,5,31)

j = jet([0.5j, np.inf, -0.5j], [2, -2])
print('q =', j.q)
print('theta =', np.degrees(j.theta))
j.plot(phi, n_stream, 'b')
j.plot_boundary('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2,2,-1.5,1.5])
plt.tight_layout()
plt.savefig('fig8a.eps')
plt.show()

plt.close()
#######################################################

plt.figure(figsize=(6.4,4.8))

n_stream = 10
phi = np.linspace(-1,5,31)

j = jet([0.5j, np.inf, -0.5j], [3/2, -1])
print('q =', j.q)
print('theta =', np.degrees(j.theta))
j.plot(phi, n_stream, 'b')
j.plot_boundary('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2,2,-1.5,1.5])
plt.tight_layout()
plt.savefig('fig8b.eps')
plt.show()

plt.close()
#######################################################

plt.figure(figsize=(6.4,3.2))

n_stream = 10
phi = np.linspace(-1.2,4.8,31)

j = jet([0.5j, 1j, np.inf, -1j, -0.5j], [1/2, 0])
print('q =', j.q)
print('theta =', np.degrees(j.theta))
j.plot(phi, n_stream, 'b')
j.plot_boundary('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2.04,2.04,-1.02,1.02])
plt.tight_layout()
plt.savefig('fig8c.eps')
plt.show()

plt.close()
#######################################################

plt.figure(figsize=(6.4,3.2))

n_stream = 10
phi = np.linspace(-2.2,4.8,36)

j = jet([1j, 2j, np.inf, 0], [1/2, 0], method='broyden2')
print('q =', j.q)
print('theta =', np.degrees(j.theta))
j.plot(phi, n_stream, 'b')
j.plot_boundary('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-3.02,3.02,-1,2.02])
plt.tight_layout()
plt.savefig('fig8d.eps')
plt.show()

plt.close()
#######################################################

plt.figure(figsize=(6.3,4.2))

n_stream = 10
phi = np.linspace(-2.4,3.8,32)

j = jet([-1, 0, 2j, np.inf, -2], [1/2, 0])
print('q =', j.q)
print('theta =', np.degrees(j.theta))
j.plot(phi, n_stream, 'b')
j.plot_boundary('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-6.01,0.02,-2,2.02])
plt.tight_layout()
plt.savefig('fig8e.eps')
plt.show()

plt.close()
#######################################################

plt.figure(figsize=(6.3,4.2))

n_stream = 10
phi = np.linspace(-8,-1.2,34)

j = jet([1j, np.inf, 0, -6], [1,0], n_node=16)
print('q =', j.q)
print('theta =', np.degrees(j.theta))
j.plot(phi, n_stream, 'b')
j.plot_boundary('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-3,1,-1.98,1.02])
plt.tight_layout()
plt.savefig('fig8f.eps')
plt.show()

plt.close()
#######################################################

plt.figure(figsize=(6.4,4.8))

n_stream = 10
phi = np.linspace(-0.4,6.4,35)

a = 0.5*np.exp(np.pi/6 * 1j)
j = jet([0.5j - np.conj(a), 0.5j, 1+0.5j, np.inf,
         1-0.5j, -0.5j, -0.5j - a],
        [1/2,-1])
print('q =', j.q)
print('theta =', np.degrees(j.theta))
j.plot(phi, n_stream, 'b')
j.plot_boundary('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-4,4,-3,3])
plt.tight_layout()
plt.savefig('fig8g.eps')
plt.show()

plt.close()
#######################################################

plt.figure(figsize=(6.4,4.8))

n_stream = 10
phi = np.linspace(-4.4,9.2,69)

j = jet([1j, 2j, np.inf, 2, 2+1j], [3/2,-1])
print('q =', j.q)
print('theta =', np.degrees(j.theta))
j.plot(phi, n_stream, 'b')
j.plot_boundary('r')
plt.axis('equal')
plt.axis('off')
plt.axis([-4,4,-0.02,5.98])
plt.tight_layout()
plt.savefig('fig8h.eps')
plt.show()
