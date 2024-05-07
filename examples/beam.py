import jax
import jax.numpy as jnp
from spinn import spinn, hcube, mlgraph


class beam(spinn):
    """
    Solve fourth order Euler-Bernoulli beam equation

        dFxxxx = q(x)

    subject to:

        F = 0       at  x=0, x=L
        dFx = 0     at  x=0, x=L

    via *unconstrained multi-objective optimization*.
    """

    @jax.jit
    def loss(self, w, x, q, bx, lmbd):
        self.w = jnp.asarray(w)

        F, dFx, dFxxxx = self.propx([[bx, 0],
                                     [bx, 0, 0],
                                     [x,  0, 0, 0, 0, 0]])

        e1 = self.mse(dFxxxx, q)
        e2 = self.mse(F,      0)
        e3 = self.mse(dFx,    0)
        l1, l2, l3 = lmbd
        return l1*e1 + l2*e2 + l3*e3


if __name__ == "__main__":
    L = 4.
    x = hcube(L, n=201)
    q = -8 * jnp.sin(jnp.pi*(x/L))
    bx = x[jnp.asarray([0, -1])]  # boundary
    lmbd = [1., 1., 1.]

    G = mlgraph((1, 32, 1))
    B = beam(G, initx=x)
    B.train(x, q, bx, lmbd, tol=1e-4, maxiter=5000, disp=True)

    # Plot
    import pylab
    F, dFx, dFxx, dFxxx, dFxxxx = B.propx([[x, 0],
                                           [x, 0, 0],
                                           [x, 0, 0, 0],
                                           [x, 0, 0, 0, 0],
                                           [x, 0, 0, 0, 0, 0]])

    pylab.plot(x, F, label='F')
    pylab.plot(x, dFx, label='dFx')
    pylab.plot(x, dFxx, label='dFxx')
    pylab.plot(x, dFxxx, label='dFxxx')
    pylab.plot(x, dFxxxx, label='dFxxxx')
    pylab.grid()
    pylab.legend()
    pylab.show()
