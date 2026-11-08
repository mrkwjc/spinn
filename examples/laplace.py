import jax
import jax.numpy as jnp
from spinn import spinn, hcube, left, right, bottom, top, mlgraph


class laplace(spinn):
    """
    Solve Laplace equation:

        dFxx + dFyy = 0

    subject to:

        F = 0                           at  x=0         (left)
        F = B/2 * (1 - cos(pi*y/B))     at  x=L         (right)
        dFy = 0                         at  y=0, y=B    (bottom, top)

    via *unconstrained multi-objective optimization*.

    The right edge profile has zero slope at y=0 and y=B, so the boundary
    data are *compatible* at the corners (L, 0) and (L, B) with dFy = 0
    on the bottom and top edges. (A simpler choice like F = y would give
    dFy = 1 there and a singular solution at these corners.) Thus the
    solution is smooth and given exactly by:

        F(x, y) = B/2 * x/L
                - B/2 * cos(pi*y/B) * sinh(pi*x/B) / sinh(pi*L/B)
    """

    @jax.jit
    def terms(self, w, x, q, bx):
        self.w = jnp.asarray(w)

        left, right, bottom, top = bx

        dFxx, dFyy = self.propx([[x, 0, 0, 0],
                                 [x, 0, 1, 1]])

        F_left, F_right = self.propx([[left,  0],
                                      [right, 0]])
        dFy_bottom, dFy_top = self.propx([[bottom, 0, 1],
                                          [top,    0, 1]])

        e1 = self.mse(dFxx + dFyy, 0)
        e2 = self.mse(F_left,      0)
        e3 = self.mse(F_right,     q)
        e4 = self.mse(dFy_bottom,  0)
        e5 = self.mse(dFy_top,     0)

        return e1, e2, e3, e4, e5


def exact(x, L, B):
    X, Y = x[:, 0], x[:, 1]
    k = jnp.pi / B
    return B/2 * X/L - B/2 * jnp.cos(k*Y) * jnp.sinh(k*X) / jnp.sinh(k*L)


if __name__ == "__main__":
    n = 21
    L = 1.
    B = 1.
    x = hcube((L, B), n=n)
    bx = left(x), right(x), bottom(x), top(x)
    q = B/2 * (1 - jnp.cos(jnp.pi * bx[1][:, 1] / B))
    # Fixed weights of the terms: the Dirichlet conditions (left, right)
    # are weighted up; automatic balancing (the default) leaves all weights
    # at 1 here, because the boundary terms already pull harder than the
    # equation residual (about 5x less accurate)
    lmbd = [1., 10., 10., 1., 1.]

    G = mlgraph((2, 16, 16, 1))
    PDE = laplace(G, initx=x)
    PDE.train(x, q, bx, tol=1e-10, maxiter=5000,
              balance=lmbd)

    # Compare with theoretical solution on the training grid...
    F = PDE(x).T[0]
    Fe = exact(x, L, B)
    print('Training grid (%dx%d):' % (n, n))
    print('  RMSE vs exact:    ', spinn.rmse(F, Fe))
    print('  Max abs error:    ', jnp.abs(F - Fe).max())
    print('  Relative L2 error:', jnp.linalg.norm(F - Fe) / jnp.linalg.norm(Fe))

    # ...and on a finer grid (points not seen during training)
    m = 101
    xt = hcube((L, B), n=m)
    Ft = PDE(xt).T[0]
    Fte = exact(xt, L, B)
    print('Test grid (%dx%d):' % (m, m))
    print('  RMSE vs exact:    ', spinn.rmse(Ft, Fte))
    print('  Max abs error:    ', jnp.abs(Ft - Fte).max())
    print('  Relative L2 error:', jnp.linalg.norm(Ft - Fte) / jnp.linalg.norm(Fte))

    #
    # Plots
    import matplotlib.pyplot as plt

    X = xt.T[0].reshape(m, m)
    Y = xt.T[1].reshape(m, m)
    Ft = Ft.reshape(m, m)
    Fte = Fte.reshape(m, m)

    fig = plt.figure(figsize=(15, 5), layout='constrained')

    # common color scale for both solutions
    norm = plt.Normalize(vmin=min(Ft.min(), Fte.min()),
                         vmax=max(Ft.max(), Fte.max()))

    ax1 = fig.add_subplot(131, projection='3d')
    s = ax1.plot_surface(X, Y, Ft, cmap='viridis', norm=norm)
    ax1.view_init(30, 225)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('PINN solution')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Fte, cmap='viridis', norm=norm)
    ax2.view_init(30, 225)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Exact solution')

    fig.colorbar(s, ax=[ax1, ax2], shrink=0.6)

    ax = fig.add_subplot(133)
    c = ax.contourf(X, Y, Ft - Fte, levels=21, cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Error: PINN - exact')

    plt.show()
