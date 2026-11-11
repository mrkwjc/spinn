# spinn - Simple Physics Informed Neural Networks
Makes PINNs easy (with jax), see examples...

## Usage

Subclass `spinn` and define `terms`: the loss terms of the problem, with the
equation (PDE residual) first and the conditions after it. `propx` returns
the network outputs and their derivatives at given points: `[x, k, i, j, ...]`
is output `k` at the points `x`, differentiated with respect to inputs
`i`, `j`, ...

```python
import jax
import jax.numpy as jnp
from spinn import spinn, hcube, left, right, bottom, mlgraph


class heat(spinn):
    """dF/dt = a d2F/dx2, F = 0 at x = 0 and x = 1, F = sin(pi x) at t = 0"""

    @jax.jit
    def terms(self, w, x, a, f0, bx):
        self.w = jnp.asarray(w)
        left, right, bottom = bx
        dFt, dFxx = self.propx([[x, 0, 1], [x, 0, 0, 0]])
        Fl, Fr, Fb = self.propx([[left, 0], [right, 0], [bottom, 0]])
        return (self.mse(dFt - a*dFxx),             # equation
                self.mse(Fl) + self.mse(Fr),        # boundary conditions
                self.mse(Fb, f0))                   # initial condition


x = hcube((1., 0.5), n=21)                  # grid of points (x, t)
bx = left(x), right(x), bottom(x)
f0 = jnp.sin(jnp.pi*bx[2][:, 0])

P = heat(mlgraph((2, 16, 16, 1)), initx=x)  # 2 inputs, 2 hidden layers
res = P.train(x, 0.2, f0, bx)
F = P(x)                                    # solution at the points
```

`train` minimizes the weighted sum of the terms with scipy's L-BFGS-B. The
weights are balanced automatically (`balance='curv'` by default, see
`spinn.balance` for the other rules, e.g. `balance='curvsens'`), or given
explicitly as `balance=[1., 10., 10.]`. The final weights are returned in
`res.lmbd` and the unweighted terms in `res.terms`.

More problems (beams, Laplace and heat equations) are in `examples/`.
