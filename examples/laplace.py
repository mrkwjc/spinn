import jax
import jax.numpy as jnp
from spinn import spinn, hcube, left, right, bottom, top, mlgraph


class laplace(spinn):
    """
    Solve Laplace equation:

        dFxx + dFyy = 0

    subject to:

        F = 0       at  x=0         (left)
        F = y       at  x=L         (right)
        dFy = 0     at  y=0, y=B    (bottom, top)

    via *unconstrained multi-objective optimization*.
    """

    @jax.jit
    def loss(self, w, x, q, bx, lmbd):
        self.w = jnp.asarray(w)

        left, right, bottom, top = bx
        l1, l2, l3, l4, l5 = lmbd

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

        return l1*e1 + l2*e2 + l3*e3 + l4*e4 + l5*e5


if __name__ == "__main__":
    n = 21
    L = 1.
    B = 2.
    x = hcube((L, B), n=n)
    bx = left(x), right(x), bottom(x), top(x)
    q = jnp.linspace(0., B, n)
    lmbd = [1., 1., 100., 1., 1.]

    G = mlgraph((2, 16, 16, 1))
    PDE = laplace(G, initx=x)
    PDE.train(x, q, bx, lmbd, tol=1e-5, maxiter=5000, disp=True)

    #
    # Plots
    import matplotlib.pyplot as plt
    from matplotlib import cm

    def plot2D(x, y, F):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, F,  cmap=cm.viridis)
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)
        ax.view_init(30, 225)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.show()

    plot2D(x.T[0].reshape(n, n),
           x.T[1].reshape(n, n),
           PDE(x).T[0].reshape(n, n))

