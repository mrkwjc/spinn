import jax
import jax.numpy as jnp
from spinn import spinn, hcube, mlgraph


class cantilever(spinn):
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

    written as a single equation with the Fischer-Burmeister function:

        g + R - sqrt(g**2 + R**2) = 0

    via *unconstrained multi-objective optimization* (penalty method).

    The exact solution is the free cantilever deflection plus the deflection
    caused by the tip force R (if in contact):

        F(x) = -c*sin(k*x) + c*k*x - c*k**3*L/2 * x**2 + c*k**3/6 * x**3
               + R * x**2 * (3*L - x) / (6*EI)

    with k = pi/L, c = q0/(EI*k**4). Without the support the tip would deflect to
    F(L) = c*(pi - pi**3/3) (about -151.2 for q0 = 8, L = 4, EI = 1); for Fmin above
    that R = 3*EI*(Fmin - F(L))/L**3, otherwise R = 0.
    """

    @jax.jit
    def terms(self, w, x, q, bx, EI, Fmin):
        self.w = jnp.asarray(w)

        F, dFx, dFxx, dFxxx = self.propx([[bx, 0],
                                          [bx, 0, 0],
                                          [bx, 0, 0, 0],
                                          [bx, 0, 0, 0, 0]])
        dFxxxx, = self.propx([[x, 0, 0, 0, 0, 0]])

        # bx = (0, L): clamped at x=0, tip at x=L
        g = F[1] - Fmin     # gap between the tip and the support
        R = -EI*dFxxx[1]    # support force at the tip
        contact = g + R - jnp.sqrt(g**2 + R**2)  # Fischer-Burmeister

        e1 = self.mse(EI*dFxxxx, q)
        e2 = self.mse(F[0],    0)
        e3 = self.mse(dFx[0],  0)
        e4 = self.mse(dFxx[1], 0)
        e5 = self.mse(contact, 0)
        return e1, e2, e3, e4, e5


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
    EI = 1.     # bending stiffness
    Fmin = 10.  # height of the tip support (try e.g. Fmin = -200, no contact)
    x = hcube(L, n=201)
    q = -q0 * jnp.sin(jnp.pi*(x/L))
    bx = x[jnp.asarray([0, -1])]  # boundary
    # Fixed weights of the terms: automatic balancing (the default) weights
    # up the boundary terms as they get satisfied, while here the equation
    # residual should be weighted up (about 2x more accurate)
    lmbd = [10., 1., 1., 1., 1.]

    G = mlgraph((1, 32, 1))
    B = cantilever(G, initx=x)
    B.train(x, q, bx, EI, Fmin, tol=1e-10, maxiter=10000,
            balance=lmbd)

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
