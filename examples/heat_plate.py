import jax
import jax.numpy as jnp
from spinn import spinn, hcube, left, right, bottom, top, mlgraph


class plate(spinn):
    """
    Steady heat conduction in a rectangular aluminium plate, in physical
    units (m, K, W):

        Txx + Tyy = 0                       0 < x < L, 0 < y < B

    subject to:

        T = T0                              at  x=0         (cooled edge)
        k*Tx = q(y)                         at  x=L         (heated edge)
        Ty = 0                              at  y=0, y=B    (insulated)

    with the heat flux q(y) = q0/2 * (1 - cos(pi*y/B)), which vanishes at
    the corners, so the boundary data are compatible and the solution is
    smooth:

        T(x, y) = T0 + a*x - a*B/pi * cos(pi*y/B) * sinh(pi*x/B)/cosh(pi*L/B)

    with a = q0/(2*k).

    In physical units the loss terms differ by orders of magnitude (the
    residual is in K/m**2, the cooled edge in K, the heat flux in W/m**2),
    so with equal lambdas the network does not find the solution at all.
    Automatic balancing of the terms (the default in 'train') weights them
    to equal curvature and gives about 0.1% error in float32 within
    seconds (about 1% with balance='grad'). Nondimensional variables (e.g.
    x/L, y/L, (T - T0)/(q0*L/k)) avoid the problem altogether, they bring
    all terms to order one.
    """

    @jax.jit
    def terms(self, w, x, bx, k, T0, q):
        self.w = jnp.asarray(w)

        left, right, bottom, top = bx

        Txx, Tyy = self.propx([[x, 0, 0, 0],
                               [x, 0, 1, 1]])
        T_left, = self.propx([[left, 0]])
        Tx_right, = self.propx([[right, 0, 0]])
        Ty_bottom, Ty_top = self.propx([[bottom, 0, 1],
                                        [top,    0, 1]])

        e1 = self.mse(Txx + Tyy,   0)
        e2 = self.mse(T_left,      T0)
        e3 = self.mse(k*Tx_right,  q)
        e4 = self.mse(Ty_bottom,   0)
        e5 = self.mse(Ty_top,      0)

        return e1, e2, e3, e4, e5


def exact(x, L, B, k, T0, q0):
    X, Y = x[:, 0], x[:, 1]
    a = q0 / (2*k)
    return (T0 + a*X
            - a*B/jnp.pi * jnp.cos(jnp.pi*Y/B)
            * jnp.sinh(jnp.pi*X/B) / jnp.cosh(jnp.pi*L/B))


if __name__ == "__main__":
    L = 0.2      # plate length [m]
    B = 0.1      # plate width [m]
    k = 200.     # thermal conductivity of aluminium [W/(m K)]
    T0 = 20.     # temperature of the cooled edge [C]
    q0 = 2e4     # maximum heat flux at the heated edge [W/m**2] (2 W/cm**2)

    n = 21
    x = hcube((L, B), n=n)
    bx = left(x), right(x), bottom(x), top(x)
    q = q0/2 * (1 - jnp.cos(jnp.pi * bx[1][:, 1] / B))

    G = mlgraph((2, 16, 16, 1))
    P = plate(G, initx=x)
    res = P.train(x, bx, k, T0, q)  # lambdas (term weights) balanced
    print('Lambdas (term weights):', res.lmbd)

    # Compare with the exact solution on a finer grid
    m = 101
    xt = hcube((L, B), n=m)
    T = P(xt).T[0]
    Te = exact(xt, L, B, k, T0, q0)
    print('Temperature range:    %.2f .. %.2f C' % (Te.min(), Te.max()))
    print('Max abs error:        %.3f K' % jnp.abs(T - Te).max())
    print('Relative L2 error of the temperature rise:',
          jnp.linalg.norm(T - Te) / jnp.linalg.norm(Te - T0))

    #
    # Plots
    import matplotlib.pyplot as plt

    X = xt.T[0].reshape(m, m)
    Y = xt.T[1].reshape(m, m)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    c = axs[0].contourf(X, Y, T.reshape(m, m), levels=21, cmap='inferno')
    fig.colorbar(c, ax=axs[0], label='T [C]')
    axs[0].set_title('PINN temperature')
    c = axs[1].contourf(X, Y, (T - Te).reshape(m, m), levels=21,
                        cmap='coolwarm')
    fig.colorbar(c, ax=axs[1], label='K')
    axs[1].set_title('Error: PINN - exact')
    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
    plt.show()
