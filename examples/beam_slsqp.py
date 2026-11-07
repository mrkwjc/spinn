import jax
import jax.numpy as jnp
from spinn import spinn, hcube, mlgraph

"""
Solve fourth order Euler-Bernoulli beam equation
    dFxxxx = q(x)
subject to:
    F = 0       at  x=0, x=L
    dFx = 0     at  x=0, x=L
via **constrained optimization**.
"""


class beam(spinn):
    @jax.jit
    def loss(self, w, x, q, bx):
        self.w = jnp.asarray(w)
        dFxxxx, = self.propx([[x, 0, 0, 0, 0, 0]])
        # The loss is scaled up, because at initial weights its gradient
        # is tiny and SLSQP tends to stop prematurely otherwise
        return 100. * self.mse(dFxxxx, q)

    @jax.jit
    def eqc(self, w, x, q, bx):
        self.w = jnp.asarray(w)
        F, dFx = self.propx([[bx, 0],
                             [bx, 0, 0]])
        return jnp.concatenate((F, dFx)).ravel()

    # @jax.jit
    # def eqc(self, w, x, q, bx):
    #     self.w = jnp.asarray(w)
    #     F, dFx, dFxx = self.propx([[bx, 0],
    #                                [bx, 0, 0],
    #                                [bx, 0, 0, 0]])
    #     bcs = [F[0], dFxx[0], dFx[1], dFxx[1]]
    #     return jnp.concatenate((bcs)).ravel()
    #
    # @jax.jit
    # def ineqc(self, w, x, q, bx):
    #     self.w = jnp.asarray(w)
    #     F, dFx = self.propx([[bx, 0],
    #                          [bx, 0, 0]])
    #     bcs = [F[1] + 6, -F[1]]
    #     return jnp.concatenate(bcs).ravel()


if __name__ == "__main__":
    L = 4.
    x = hcube(L, n=201)
    q = -8. * jnp.sin(jnp.pi*(x/L))
    bx = x[jnp.asarray([0, -1])]  # boundary

    G = mlgraph([1, 32, 1], full=True)
    B = beam(G, initx=x)
    B.train(x, q, bx, tol=1e-8, maxiter=5000, disp=True)

    F, dFx, dFxx, dFxxx, dFxxxx = B.propx([[x, 0],
                                           [x, 0, 0],
                                           [x, 0, 0, 0],
                                           [x, 0, 0, 0, 0],
                                           [x, 0, 0, 0, 0, 0]])

    # Plot
    import pylab
    pylab.plot(x, F, label='F')
    pylab.plot(x, dFx, label='dFx')
    pylab.plot(x, dFxx, label='dFxx')
    pylab.plot(x, dFxxx, label='dFxxx')
    pylab.plot(x, dFxxxx, label='dFxxxx')
    pylab.grid()
    pylab.legend()
    pylab.show()
