import numpy as np
import matplotlib.pyplot as plt

dt = 0.01                          
omega = np.linspace(0, 1/dt, 500)[1:]  

# critical dampening (divergence boundary)
p_omega = (-1 + np.sqrt(1 - (dt*omega)**2)) / dt

plt.plot(omega, p_omega)
plt.axhline(0, color='k', linewidth=.5) 
plt.xlabel('omega  [rad s⁻¹]')
plt.ylabel('p_omega  [s⁻¹]')
plt.title('Divergence boundary for dt = %.3f s' % dt)
plt.show()
