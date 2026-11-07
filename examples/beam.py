import jax
import jax.numpy as jnp
from spinn import spinn, hcube, mlgraph


class beam(spinn):
    """
    Solve fourth order Euler-Bernoulli beam equation

        dFxxxx = q(x),      q(x) = -q0 * sin(pi*x/L)

    subject to:

        F = 0       at  x=0, x=L
        dFx = 0     at  x=0, x=L

    (beam clamped at both ends) via *unconstrained multi-objective
    optimization*.

    With k = pi/L and c = q0/k**4 the exact solution is:

        F(x) = -c * (sin(k*x) - k*x*(L - x)/L)

    i.e. the particular solution -c*sin(k*x) corrected by a polynomial
    which restores zero slopes at both ends.
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


def exact(x, q0, L):
    """Exact F and its derivatives up to the 4th order."""
    x = x.ravel()
    k = jnp.pi / L
    c = q0 / k**4
    s, co = jnp.sin(k*x), jnp.cos(k*x)
    F = -c * (s - k*x*(L - x)/L)
    dFx = -c * (k*co - k*(L - 2*x)/L)
    dFxx = -c * (-k**2*s + 2*k/L)
    dFxxx = c * k**3 * co
    dFxxxx = -q0 * s
    return F, dFx, dFxx, dFxxx, dFxxxx


if __name__ == "__main__":
    L = 4.
    q0 = 8.
    x = hcube(L, n=401)
    q = -q0 * jnp.sin(jnp.pi*(x/L))
    bx = x[jnp.asarray([0, -1])]  # boundary
    lmbd = [10., 1., 1.]

    G = mlgraph((1, 32, 1))
    B = beam(G, initx=x)
    B.train(x, q, bx, lmbd, tol=1e-8, maxiter=5000)

    # Compare with theoretical solution
    names = ['F', 'dFx', 'dFxx', 'dFxxx', 'dFxxxx']
    pinn = B.propx([[x, 0],
                    [x, 0, 0],
                    [x, 0, 0, 0],
                    [x, 0, 0, 0, 0],
                    [x, 0, 0, 0, 0, 0]])
    theory = exact(x, q0, L)

    print('%-7s %12s %12s %12s' % ('', 'RMSE', 'Max abs err', 'Rel. L2 err'))
    for name, Fp, Fe in zip(names, pinn, theory):
        print('%-7s %12.3e %12.3e %12.3e' % (
            name,
            spinn.rmse(Fp, Fe),
            jnp.abs(Fp - Fe).max(),
            jnp.linalg.norm(Fp - Fe) / jnp.linalg.norm(Fe)))

    #
    # Plots
    import matplotlib.pyplot as plt

    xr = x.ravel()
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for ax, name, Fp, Fe in zip(axs.flat, names, pinn, theory):
        ax.plot(xr, Fe, 'k--', lw=2, label='exact')
        ax.plot(xr, Fp, label='PINN')
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.grid()
        ax.legend()

    ax = axs.flat[-1]
    for name, Fp, Fe in zip(names, pinn, theory):
        ax.semilogy(xr, jnp.abs(Fp - Fe), label=name)
    ax.set_title('|PINN - exact|')
    ax.set_xlabel('x')
    ax.grid()
    ax.legend()

    plt.show()
