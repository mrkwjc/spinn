import jax
import jax.numpy as jnp
from spinn import spinn, hcube, mlgraph

# SLSQP gains much accuracy in 64-bit (in float32 it stops early, near
# the float32 rounding level)
jax.config.update("jax_enable_x64", True)

"""
Solve fourth order Euler-Bernoulli equation for a cantilever beam whose
tip may rest on a support at height Fmin:

    EI * dFxxxx = q(x),      q(x) = -q0 * sin(pi*x/L)

where EI is the bending stiffness, subject to:

    F = 0           at  x=0     (clamped end)
    dFx = 0         at  x=0
    dFxx = 0        at  x=L     (no bending moment at the tip)

and the contact condition at the tip, with the gap g = F(L) - Fmin and
the support force R = -EI*dFxxx(L):

    g >= 0,  R >= 0,  g*R = 0

i.e. the tip cannot go below the support, the support can only push,
and it pushes only when the tip touches it (otherwise dFxxx(L) = 0, the
free end). All solved via **constrained optimization**.

Writing the contact condition directly as eqc/ineqc makes SLSQP fail
often (the constraints become linearly dependent at contact). Instead it
is imposed as a single equality with the Fischer-Burmeister function:

    g + R - sqrt(g**2 + R**2) = 0   <=>   g >= 0, R >= 0, g*R = 0

The exact solution is the free cantilever deflection plus the deflection
caused by the tip force R (if in contact):

    F(x) = -c*sin(k*x) + c*k*x - c*k**3*L/2 * x**2 + c*k**3/6 * x**3
           + R * x**2 * (3*L - x) / (6*EI)

with k = pi/L, c = q0/(EI*k**4). Without the support the tip would deflect to
F(L) = c*(pi - pi**3/3) (about -151.2 for q0 = 8, L = 4, EI = 1); for Fmin above
that R = 3*EI*(Fmin - F(L))/L**3, otherwise R = 0.
"""

EI = 1.      # bending stiffness
Fmin = -30.  # height of the tip support (try e.g. Fmin = -200, no contact)


class beam(spinn):
    @jax.jit
    def loss(self, w, x, q, bx):
        self.w = jnp.asarray(w)
        dFxxxx, = self.propx([[x, 0, 0, 0, 0, 0]])
        # The loss is scaled up, because at initial weights its gradient
        # is tiny and SLSQP tends to stop prematurely otherwise
        return 100. * self.mse(EI*dFxxxx, q)

    @jax.jit
    def eqc(self, w, x, q, bx):
        self.w = jnp.asarray(w)
        F, dFx, dFxx, dFxxx = self.propx([[bx, 0],
                                          [bx, 0, 0],
                                          [bx, 0, 0, 0],
                                          [bx, 0, 0, 0, 0]])
        # bx = (0, L): clamped at x=0, tip at x=L
        g = F[1] - Fmin     # gap between the tip and the support
        R = -EI*dFxxx[1]    # support force at the tip
        contact = g + R - jnp.sqrt(g**2 + R**2)  # Fischer-Burmeister
        return jnp.stack([F[0], dFx[0], dFxx[1], contact])


def exact(x, q0, L, EI, Fmin):
    """Exact deflection and the tip support force."""
    x = x.ravel()
    k = jnp.pi / L
    c = q0 / (EI * k**4)
    F = (-c*jnp.sin(k*x) + c*k*x
         - c*k**3*L/2 * x**2 + c*k**3/6 * x**3)
    FL = c*(jnp.pi - jnp.pi**3/3)               # free tip deflection
    R = jnp.maximum(0., 3*EI*(Fmin - FL)/L**3)     # tip force for contact
    return F + R * x**2 * (3*L - x) / (6*EI), R


if __name__ == "__main__":
    L = 4.
    q0 = 8.
    x = hcube(L, n=201)
    q = -q0 * jnp.sin(jnp.pi*(x/L))
    bx = x[jnp.asarray([0, -1])]  # boundary

    G = mlgraph([1, 32, 1], full=True)
    B = beam(G, initx=x)
    # warm start: 100 iterations of the penalty method before SLSQP,
    # which prevents SLSQP from diverging from poor initial weights
    B.train(x, q, bx, tol=1e-8, maxiter=5000, warm=100, disp=True)

    F, dFx, dFxx, dFxxx, dFxxxx = B.propx([[x, 0],
                                           [x, 0, 0],
                                           [x, 0, 0, 0],
                                           [x, 0, 0, 0, 0],
                                           [x, 0, 0, 0, 0, 0]])

    Fe, Re = exact(x, q0, L, EI, Fmin)
    print('Tip deflection F(L):  %.6f (exact %.6f, support at %g)'
          % (F[-1], Fe[-1], Fmin))
    print('Support force R:      %.6f (exact %.6f)' % (-EI*dFxxx[-1], Re))
    print('Relative L2 error of F:', jnp.linalg.norm(F - Fe)
          / jnp.linalg.norm(Fe))

    # Plot
    import pylab
    pylab.plot(x, Fe, 'k--', lw=2, label='F exact')
    pylab.plot(x, F, label='F')
    pylab.plot(x, dFx, label='dFx')
    pylab.plot(x, dFxx, label='dFxx')
    pylab.plot(x, dFxxx, label='dFxxx')
    pylab.plot(x, dFxxxx, label='dFxxxx')
    pylab.axhline(Fmin, color='grey', ls=':', label='support')
    pylab.grid()
    pylab.legend()
    pylab.show()
