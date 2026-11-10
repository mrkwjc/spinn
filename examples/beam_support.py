import jax
import jax.numpy as jnp
from spinn import spinn, hcube, mlgraph


class beam(spinn):
    """
    Solve fourth order Euler-Bernoulli equation for a beam clamped at both
    ends, whose middle may rest on a support at height Fmin:

        EI * dFxxxx = q(x) + R*delta(x - L/2),      q(x) = -q0 * sin(pi*x/L)

    where EI is the bending stiffness, subject to:

        F = 0, dFx = 0      at  x=0 and x=L     (clamped ends)

    and the contact condition at the middle, with the gap g = F(L/2) - Fmin
    and the support force R:

        g >= 0,  R >= 0,  g*R = 0

    The support force R is a point load, so EI*dFxxx jumps by R at x=L/2.
    A single (smooth) network cannot represent this jump, thus the beam is
    split into two parts: F1 on [0, L/2] (network output 0) and F2 on
    [L/2, L] (network output 1), joined at x=L/2 by:

        F1 = F2,  dF1x = dF2x,  dF1xx = dF2xx     (deflection, slope, moment)
        R = EI*(dF2xxx - dF1xxx)                      (jump of the shear force)

    The contact condition is written with the Fischer-Burmeister function:

        g + R - sqrt(g**2 + R**2) = 0   <=>   g >= 0, R >= 0, g*R = 0

    All solved via *unconstrained multi-objective optimization* (penalty
    method).

    The exact solution is the free deflection plus the deflection caused by
    the point force R at the middle:

        F(x) = -c*(sin(k*x) - k*x*(L - x)/L) + R * x**2 * (3*L - 4*x) / (48*EI)

    for x <= L/2 (symmetric for x > L/2), with k = pi/L, c = q0/(EI*k**4).
    Without the support the middle would deflect to F(L/2) = -c*(1 - pi/4)
    (about -4.51 for q0 = 8, L = 4, EI = 1); for Fmin above that
    R = 192*EI*(Fmin - F(L/2))/L**3, otherwise R = 0.
    """

    @jax.jit
    def terms(self, w, x, q, bx, EI, Fmin):
        self.w = jnp.asarray(w)
        x1, x2 = x      # domain of each part (output)
        q1, q2 = q

        dF1xxxx, dF2xxxx = self.propx([[x1, 0, 0, 0, 0, 0],
                                       [x2, 1, 0, 0, 0, 0]])
        # bx = (0, L/2, L)
        F1, dF1x, dF1xx, dF1xxx = self.propx([[bx, 0],
                                              [bx, 0, 0],
                                              [bx, 0, 0, 0],
                                              [bx, 0, 0, 0, 0]])
        F2, dF2x, dF2xx, dF2xxx = self.propx([[bx, 1],
                                              [bx, 1, 0],
                                              [bx, 1, 0, 0],
                                              [bx, 1, 0, 0, 0]])

        g = F1[1] - Fmin            # gap between the middle and the support
        R = EI*(dF2xxx[1] - dF1xxx[1])  # support force (jump of shear)
        contact = g + R - jnp.sqrt(g**2 + R**2)  # Fischer-Burmeister

        e1 = self.mse(EI*dF1xxxx, q1) + self.mse(EI*dF2xxxx, q2)
        e2 = (F1[0]**2 + dF1x[0]**2 + F2[2]**2 + dF2x[2]**2)
        e3 = ((F1[1] - F2[1])**2 + (dF1x[1] - dF2x[1])**2
              + (dF1xx[1] - dF2xx[1])**2)
        e4 = contact**2
        return e1, e2, e3, e4


def exact(x, q0, L, EI, Fmin):
    """Exact deflection and the middle support force."""
    x = x.ravel()
    k = jnp.pi / L
    c = q0 / (EI * k**4)
    F = -c*(jnp.sin(k*x) - k*x*(L - x)/L)
    Fm = -c*(1 - jnp.pi/4)                      # free middle deflection
    R = jnp.maximum(0., 192*EI*(Fmin - Fm)/L**3)   # middle force for contact
    s = jnp.minimum(x, L - x)                   # symmetry
    return F + R * s**2 * (3*L - 4*s) / (48*EI), R


if __name__ == "__main__":
    L = 4.
    q0 = 8.
    EI = 1.     # bending stiffness
    Fmin = -4.  # height of the middle support (try e.g. Fmin = -5, no contact)
    n = 101
    x1 = hcube(L/2, n=n)                    # left part
    x2 = hcube(L/2, n=n, origin=(L/2,))     # right part
    q1 = -q0 * jnp.sin(jnp.pi*(x1/L))
    q2 = -q0 * jnp.sin(jnp.pi*(x2/L))
    bx = jnp.asarray([[0.], [L/2], [L]])    # ends and the support

    G = mlgraph((1, 32, 2))
    B = beam(G, initx=jnp.vstack([x1, x2]))
    # weights of the terms are balanced automatically (balance='curv')
    B.train((x1, x2), (q1, q2), bx, EI, Fmin, tol=1e-10, maxiter=10000)

    # Solution assembled from both parts
    x = jnp.vstack([x1, x2[1:]])
    d = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]
    P1 = B.propx([[x1, 0] + s[1:] for s in d])
    P2 = B.propx([[x2[1:], 1] + s[1:] for s in d])
    F, dFx, dFxx, dFxxx, dFxxxx = [jnp.concatenate(p) for p in zip(P1, P2)]
    m = n - 1   # index of x = L/2 in x
    R = EI*(B.propx([[bx[1:2], 1, 0, 0, 0]])[0][0] - dFxxx[m])

    Fe, Re = exact(x, q0, L, EI, Fmin)
    print('Middle deflection F(L/2):  %.6f (exact %.6f, support at %g)'
          % (F[m], Fe[m], Fmin))
    print('Support force R:           %.6f (exact %.6f)' % (R, Re))
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
    pylab.plot([L/2], [Fmin], 'k^', ms=10, label='support')
    pylab.grid()
    pylab.legend()
    pylab.show()
