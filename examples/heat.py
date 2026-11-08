import jax
import jax.numpy as jnp
from spinn import spinn, hcube, left, right, bottom, mlgraph


class heat(spinn):
    """
    Solve 1D heat (diffusion) equation:

        dFt = a * dFxx          for  0 < x < L,  0 < t < T

    subject to:

        F = 0               at  x=0, x=L    (left, right)
        F = sin(pi*x/L)     at  t=0         (bottom)

    via *unconstrained multi-objective optimization*.

    The exact solution is:

        F(x, t) = exp(-a*(pi/L)**2 * t) * sin(pi*x/L)

    Inputs are ordered as (x, t), so derivative index 0 is d/dx
    and index 1 is d/dt.
    """

    @jax.jit
    def terms(self, w, x, a, f0, bx):
        self.w = jnp.asarray(w)

        left, right, bottom = bx

        dFt, dFxx = self.propx([[x, 0, 1],
                                [x, 0, 0, 0]])

        F_left, F_right, F_bottom = self.propx([[left,   0],
                                                [right,  0],
                                                [bottom, 0]])

        e1 = self.mse(dFt - a*dFxx, 0)
        e2 = self.mse(F_left, 0) + self.mse(F_right, 0)
        e3 = self.mse(F_bottom, f0)

        return e1, e2, e3


def exact(x, a, L):
    return jnp.exp(-a*(jnp.pi/L)**2 * x[:, 1]) * jnp.sin(jnp.pi*x[:, 0]/L)


if __name__ == "__main__":
    n = 21
    L = 1.      # rod length
    T = 0.5     # final time
    a = 0.2     # diffusivity
    x = hcube((L, T), n=n)
    bx = left(x), right(x), bottom(x)
    f0 = jnp.sin(jnp.pi*bx[2][:, 0]/L)  # initial temperature profile

    G = mlgraph((2, 16, 16, 1))
    PDE = heat(G, initx=x)
    PDE.train(x, a, f0, bx)  # weights of the terms balanced automatically

    F = PDE(x).T[0]
    Fe = exact(x, a, L)
    print('RMSE vs exact solution:', spinn.rmse(F, Fe))
    print('Max abs error:         ', jnp.abs(F - Fe).max())

    #
    # Plots
    import matplotlib.pyplot as plt

    X = x.T[0].reshape(n, n)
    Tm = x.T[1].reshape(n, n)
    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Tm, F.reshape(n, n), cmap='inferno')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('PINN solution F(x, t)')
    ax.view_init(30, 225)

    ax = fig.add_subplot(122)
    for i in range(0, n, 5):  # temperature profiles at selected times
        ti = Tm[i, 0]
        line, = ax.plot(X[i], F.reshape(n, n)[i], label=f't={ti:.2f}')
        ax.plot(X[i], Fe.reshape(n, n)[i], 'o', color=line.get_color(),
                markersize=3)
    ax.set_xlabel('x')
    ax.set_ylabel('F')
    ax.set_title('PINN (lines) vs exact (dots)')
    ax.grid()
    ax.legend()

    plt.tight_layout()
    plt.show()
