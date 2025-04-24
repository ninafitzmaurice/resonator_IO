import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

dt = 0.01                          
omega = np.linspace(0, 1/dt, 500)[1:]  

# critical dampening (divergence boundary)
p_omega = (-1 + np.sqrt(1 - (dt*omega)**2)) / dt

# plt.plot(omega, p_omega)
# plt.axhline(0, color='k', linewidth=.5) 
# plt.xlabel('omega ')
# plt.ylabel('p_omega')
# plt.title('Divergence boundary for dt = %.3f s' % dt)
# plt.show()

u = sp.symbols('u', complex=True)
w, b, DT, x = sp.symbols('w b dt x', real=True)

u_next = u + DT * ((b + sp.I * w) * u + x)

U = sp.symbols('U', real=True)
V = sp.symbols('V', real=True)
def rewrite(expr):
    return expr.subs(sp.re(u),U).subs(sp.im(u), V)

print('u_next: ', rewrite(sp.re(u_next)))
print('v_next: ', rewrite(sp.im(u_next)))


