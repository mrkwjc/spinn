import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import legendre
from spinn import spinn, hcube, left, right, bottom, top, mlgraph


class plate(spinn):
    """
    Bending of a Kirchhoff-Love plate with free edges resting on an elastic
    (Winkler) subsoil:

        D*(Wxxxx + 2*Wxxyy + Wyyyy) + k*W = q      -L/2 < x < L/2,
                                                   -B/2 < y < B/2

    where W = W(x, y) is the deflection, q the surface load, k = k(x, y)
    the subsoil stiffness coefficient and D = E*H**3/(12*(1 - v**2)) the
    bending stiffness of the plate, with E being its deformation modulus,
    H its thickness and v the Poisson ratio. At the free edges the bending
    moments, effective (Kirchhoff) shear forces and corner forces vanish:

        -D*(Wxx + v*Wyy) = 0              at  x=-L/2, x=L/2   (Mx = 0)
        -D*(Wxxx + (2-v)*Wxyy) = 0        at  x=-L/2, x=L/2   (Vx = 0)
        -D*(Wyy + v*Wxx) = 0              at  y=-B/2, y=B/2   (My = 0)
        -D*(Wyyy + (2-v)*Wxxy) = 0        at  y=-B/2, y=B/2   (Vy = 0)
        -2*D*(1-v)*Wxy = 0                at  the 4 corners   (R = 0)

    For constant q and k the plate just settles uniformly, W = q/k, without
    bending. Here the subsoil is stiffer towards the edges and corners,

        k = k0 + k1*((2*x/L)**2 + (2*y/B)**2),

    so the plate bends. The subsoil should be smooth: e.g. for
    k = k0 + k1*sqrt(|x*y|) the derivatives of k are infinite at the
    axes, and so the 4th derivatives of the deflection have kinks there,
    which the network cannot follow (the equation term gets stuck about
    500 times higher than here).

    The sine activation in the hidden layers is about 30 times more
    accurate in the equation term here than the default softplus.

    The problem is solved in nondimensional variables: the coordinates are
    scaled by the characteristic length of the plate on the subsoil,
    lc = (D/k0)**(1/4), and the deflection by the settlement on the softest
    subsoil, W0 = q/k0. Then (with x, y, W denoting the scaled variables)

        Wxxxx + 2*Wxxyy + Wyyyy + (k/k0)*W = 1

    and the edge conditions keep their form, with D dropped. In physical
    units (m, kN) the deflection is of order 1e-2 m and the network gets
    stuck at a nearly flat solution (uniform settlement, no bending) even
    with automatic balancing of the terms; scaling the deflection alone
    fixes it.

    There is no closed-form solution, so the PINN is compared with a Ritz
    (energy) solution with a Legendre polynomial basis. The free edge
    conditions are natural boundary conditions of the plate energy, so
    the basis does not need to satisfy them.
    """

    @jax.jit
    def terms(self, w, x, kr, v, bx):
        self.w = jnp.asarray(w)

        left, right, bottom, top = bx
        left_right = jnp.concatenate([left, right])
        bottom_top = jnp.concatenate([bottom, top])
        corners = jnp.asarray((left[0], left[-1], right[0], right[-1]))

        W, Wxxxx, Wxxyy, Wyyyy = self.propx([[x, 0],
                                             [x, 0, 0, 0, 0, 0],
                                             [x, 0, 0, 0, 1, 1],
                                             [x, 0, 1, 1, 1, 1]])
        Wxx_lr, Wyy_lr, Wxxx, Wxyy = self.propx([[left_right, 0, 0, 0],
                                                 [left_right, 0, 1, 1],
                                                 [left_right, 0, 0, 0, 0],
                                                 [left_right, 0, 0, 1, 1]])
        Wxx_bt, Wyy_bt, Wyyy, Wxxy = self.propx([[bottom_top, 0, 0, 0],
                                                 [bottom_top, 0, 1, 1],
                                                 [bottom_top, 0, 1, 1, 1],
                                                 [bottom_top, 0, 0, 0, 1]])
        Wxy, = self.propx([[corners, 0, 0, 1]])

        e1 = self.mse(Wxxxx + 2*Wxxyy + Wyyyy + kr*W,  1)
        e2 = self.mse(Wxx_lr + v*Wyy_lr,               0)   # Mx
        e3 = self.mse(Wxxx + (2-v)*Wxyy,               0)   # Vx
        e4 = self.mse(Wyy_bt + v*Wxx_bt,               0)   # My
        e5 = self.mse(Wyyy + (2-v)*Wxxy,               0)   # Vy
        e6 = self.mse(Wxy,                             0)   # R

        return e1, e2, e3, e4, e5, e6


def subsoil(x, L, B, k0, k1):
    """Subsoil stiffness coefficient k(x, y) [kN/m**3]."""
    return k0 + k1*((2*x[:, 0]/L)**2 + (2*x[:, 1]/B)**2)


def ritz(L, B, D, v, q, k0, k1, p=20, ng=80):
    """Ritz solution with the basis P_i(2x/L)*P_j(2y/B), i, j <= p (P_i
    Legendre polynomials), integrated with ng x ng Gauss points. Returns
    the deflection W(x) as a function of points x (an (n, 2) array)."""
    t, wt = legendre.leggauss(ng)
    eye = np.eye(p + 1)

    def basis(s):  # basis values and 1st, 2nd derivatives, half length s
        return [np.stack([legendre.legval(t, legendre.legder(eye[i], d))
                          for i in range(p + 1)], 1) / s**d
                for d in range(3)]

    X0, X1, X2 = basis(L/2)
    Y0, Y1, Y2 = basis(B/2)
    xg = np.stack(np.meshgrid(t*L/2, t*B/2, indexing='ij'), -1).reshape(-1, 2)
    wg = np.outer(wt*L/2, wt*B/2)
    kg = np.asarray(subsoil(xg, L, B, k0, k1)).reshape(ng, ng)

    def M(A, C, Bx, Dy, w):  # integral of (A_a C_b) * (Bx_c Dy_d) w dA
        n = (p + 1)**2
        return np.einsum('ij,ia,jb,ic,jd->abcd', w, A, C, Bx, Dy).reshape(n, n)

    # stiffness matrix from the plate and subsoil energy, and load vector
    K = (D*(M(X2, Y0, X2, Y0, wg) + M(X0, Y2, X0, Y2, wg)
            + v*(M(X2, Y0, X0, Y2, wg) + M(X0, Y2, X2, Y0, wg))
            + 2*(1 - v)*M(X1, Y1, X1, Y1, wg))
         + M(X0, Y0, X0, Y0, wg*kg))
    f = q*np.einsum('ij,ia,jb->ab', wg, X0, Y0).ravel()
    c = np.linalg.solve(K, f).reshape(p + 1, p + 1)

    def W(x):
        x = np.asarray(x)
        return np.einsum('ia,ab,ib->i', legendre.legvander(2*x[:, 0]/L, p), c,
                         legendre.legvander(2*x[:, 1]/B, p))
    return W


if __name__ == "__main__":
    L = 10.        # plate length [m]
    B = 5.         # plate width [m]
    H = 0.2        # plate thickness [m]
    E = 35e6       # deformation modulus of concrete [kN/m**2]
    v = 0.2        # Poisson ratio of concrete
    q = 65.        # surface load [kN/m**2]
    k0 = 1000.     # subsoil stiffness at the center [kN/m**3]
    k1 = 1000.     # its growth towards the edges [kN/m**3]
    D = E*H**3 / (12*(1 - v**2))  # bending stiffness [kNm]

    lc = (D/k0)**0.25  # characteristic length [m]
    W0 = q/k0          # deflection scale [m]

    n = 31
    x = hcube((L, B), n=n, origin=(-L/2, -B/2))  # [m]
    xs = x/lc                                    # scaled
    bx = left(xs), right(xs), bottom(xs), top(xs)
    kr = subsoil(x, L, B, k0, k1)/k0

    G = mlgraph((2, 32, 32, 1))
    P = plate(G, initx=xs, f='sin')
    res = P.train(xs, kr, v, bx, monitor=True, disp=True, balance='curvsens')  # lambdas (term weights) balanced
    print('Lambdas (term weights):', res.lmbd)

    # Compare with the Ritz solution on a finer grid, in physical units
    m = 101
    xt = hcube((L, B), n=m, origin=(-L/2, -B/2))
    W = W0*P(xt/lc).T[0]
    Wr = ritz(L, B, D, v, q, k0, k1)(xt)
    print('Deflection:   PINN %.2f .. %.2f mm, Ritz %.2f .. %.2f mm'
          % (1e3*W.min(), 1e3*W.max(), 1e3*Wr.min(), 1e3*Wr.max()))
    print('Max abs error:                  %.3f mm'
          % (1e3*jnp.abs(W - Wr).max()))
    print('Relative L2 error:              %.2e'
          % (jnp.linalg.norm(W - Wr) / jnp.linalg.norm(Wr)))
    print('Relative L2 error of bending:   %.2e  (deflection minus its mean)'
          % (jnp.linalg.norm(W - Wr) / jnp.linalg.norm(Wr - Wr.mean())))

    # Bending moments [kNm/m]
    Wxx, Wyy = P.propx([[xt/lc, 0, 0, 0],
                        [xt/lc, 0, 1, 1]])
    Mx = -D*W0/lc**2 * (Wxx + v*Wyy)
    My = -D*W0/lc**2 * (Wyy + v*Wxx)
    print('Bending moments:  Mx %.2f .. %.2f, My %.2f .. %.2f kNm/m'
          % (Mx.min(), Mx.max(), My.min(), My.max()))

    #
    # Plots
    import matplotlib.pyplot as plt

    X = xt.T[0].reshape(m, m)
    Y = xt.T[1].reshape(m, m)
    fig = plt.figure(figsize=(15, 8), layout='constrained')

    ax = fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, -1e3*W.reshape(m, m), cmap='viridis')
    ax.view_init(30, 225)
    ax.set_aspect('equalxy')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('PINN deflection -W [mm]')

    panels = [(1e3*(W - Wr), 'Error: PINN - Ritz', 'mm'),
              (Mx, 'Bending moment Mx', 'kNm/m'),
              (My, 'Bending moment My', 'kNm/m')]
    for i, (F, title, unit) in enumerate(panels):
        ax = fig.add_subplot(222 + i)
        c = ax.contourf(X, Y, F.reshape(m, m), levels=21, cmap='coolwarm')
        fig.colorbar(c, ax=ax, label=unit)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(title)

    plt.show()
