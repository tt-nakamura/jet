import numpy as np
from kirch import kirch
import matplotlib.pyplot as plt

def plot_cmplx(z, *a, **k):
    plt.plot(np.real(z), np.imag(z), *a, **k)

plt.figure(figsize=(6.4,3))

phi = np.linspace(-3, 6, 91)
psi = np.linspace(-2, 2, 41)

k = kirch([-0.5j, 0.5j])
f = k.force()
print('Drag =', np.real(f))
print('Lift =', np.imag(f))
print('W =', np.abs(k.C))
k.plot(phi, psi, 'b', lw=0.5)
plot_cmplx(k.vertex, 'r')
plt.axis('equal')
plt.axis('off')
plt.axis([-3.5, 4.5, -2, 2])
plt.tight_layout()
plt.savefig('fig4a.eps')
plt.show()

plt.close()
###########################################################

plt.figure(figsize=(6.4,3))

phi = np.linspace(-1.1, 2.3, 69)
psi = np.linspace(-1, 1, 41)

k = kirch([0.5*np.exp(-np.pi/4*1j), 0, 0.5*np.exp(np.pi/6*1j)])
f = k.force()
print('Drag =', np.real(f))
print('Lift =', np.imag(f))
print('W =', np.abs(k.C))
k.plot(phi, psi, 'b', lw=0.5)
plot_cmplx(k.vertex, 'r')
plt.axis('equal')
plt.axis('off')
plt.axis([-1.2, 2, -0.8, 0.8])
plt.tight_layout()
plt.savefig('fig4b.eps')
plt.show()

plt.close()
###########################################################

plt.figure(figsize=(6.4,3))

phi = np.linspace(-1, 3.5, 91)
psi = np.linspace(-1, 1.5, 51)

a = 0.5*np.exp(-np.pi/6 * 1j)
k = kirch([a, -a, 0])
f = k.force()
print('Drag =', np.real(f))
print('Lift =', np.imag(f))
print('W =', np.abs(k.C))
k.plot(phi, psi, 'b', lw=0.5)
plot_cmplx(k.vertex, 'r')
plt.axis('equal')
plt.axis('off')
plt.axis([-1.2, 2, -0.8, 0.8])
plt.tight_layout()
plt.savefig('fig4c.eps')
plt.show()

plt.close()
###########################################################

plt.figure(figsize=(6.4,3))

phi = np.linspace(-1, 3.5, 91)
psi = np.linspace(-1, 1.5, 51)

a = 0.5*np.exp(-np.pi/6 * 1j)
k = kirch([a, -a, 0, 0.2*np.exp(np.pi/4 * 1j)])
f = k.force()
print('Drag =', np.real(f))
print('Lift =', np.imag(f))
print('W =', np.abs(k.C))
k.plot(phi, psi, 'b', lw=0.5)
plot_cmplx(k.vertex, 'r')
plt.axis('equal')
plt.axis('off')
plt.axis([-1.2, 2, -0.8, 0.8])
plt.tight_layout()
plt.savefig('fig4d.eps')
plt.show()

plt.close()
###########################################################

plt.figure(figsize=(6.4,3))

phi = np.linspace(-1.6, 4.5, 62)
psi = np.linspace(-1.6, 1.6, 33)

k = kirch([3**0.5/2, -0.5j, 0.5j, 3**0.5/2])
f = k.force()
print('Drag =', np.real(f))
print('Lift =', np.imag(f))
print('W =', np.abs(k.C))
k.plot(phi, psi, 'b', lw=0.5)
plot_cmplx(k.vertex, 'r')
plt.axis('equal')
plt.axis('off')
plt.axis([-1.5, 2.5, -1, 1])
plt.tight_layout()
plt.savefig('fig4e.eps')
plt.show()

plt.close()
###########################################################

plt.figure(figsize=(6.4,3))

phi = np.linspace(-2.4, 10, 63)
psi = np.linspace(-2.6, 3.6, 32)

a = 0.5/np.sin(np.pi/5)
k = kirch([a*np.exp(-2*np.pi*(t+0.125)*1j/5) for t in range(6)])
f = k.force()
print('Drag =', np.real(f))
print('Lift =', np.imag(f))
print('W =', np.abs(k.C))
k.plot(phi, psi, 'b', lw=0.5)
plot_cmplx(k.vertex, 'r')
plt.axis('equal')
plt.axis('off')
plt.axis([-2.5, 3.5, -1.5, 1.5])
plt.tight_layout()
plt.savefig('fig4f.eps')
plt.show()
