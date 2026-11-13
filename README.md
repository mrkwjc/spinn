# spinn - Simple Physics Informed Neural Networks
Makes PINNs easy (with jax), see examples...

## Installation

```
pip install spinn
```

spinn runs on the CPU as installed; for an NVIDIA GPU install jax with CUDA
first (`pip install -U "jax[cuda12]"`, see the jax documentation). Training
runs on the GPU without copies to the host, so it pays off there.

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

`train` minimizes the weighted sum of the terms. The weights of the terms,
the lambdas, are balanced automatically (`balance='curv'` by default, see
`spinn.balance` for the other rules), or given explicitly as
`balance=[1., 10., 10.]`. The final lambdas are returned in `res.lmbd` and
the unweighted terms in `res.terms`.

The optimizer is L-BFGS of optax by default (`method='lbfgs'`), run on the
GPU or CPU without copies to the host. `method='adam'` uses Adam (30000
iterations by default; slower, sometimes more accurate), and scipy methods
are available by name, e.g. `method='L-BFGS-B'`. Optimizers with own options
are instances (`from spinn import optimize`, then
`method=optimize.adam(lr=1e-2)`); see `help(optimize)`.

Activations of the hidden layers are set with `f`: a name from `jax.nn` or
`jax.numpy` (`f='tanh'`, `f='sin'`, `f='gelu'`; softplus by default), a
function, or one per layer. Networks are saved with `P.dump(fname)` and
restored with `heat.load(fname)`; see `help(spinn)` for details.

The course of a training is recorded by a monitor:

```python
from spinn import monitor
m = monitor(every=100, metrics={'err': lambda P: error(P)})
P.train(x, 0.2, f0, bx, monitor=m)
m.history['terms']    # numpy arrays, one row per record
m.plot()
```

A training is continued with `res = P.resume(res, maxiter=5000)`: from where
it stopped, with the state of its balancer and optimizer.

## Accuracy

Two options lower the error at a higher cost:

- `balance='curvsens'`: curvature lambdas corrected by the sensitivity of
  the solution to each term; often more accurate, at about 1.5 times the
  cost.
- `seeds=3`: training from three random initial weights, the best is kept
  (`res.seed` reproduces it); lowered the median error 1.3-5 times on the
  examples and removed most bad runs, at 3 times the cost.

## Examples

More problems are in `examples/`: beams (with a support, and a cantilever),
the heat equation, a heated plate in physical units and the Laplace
equation, all compared with exact solutions; `monitor.py` records and
compares trainings of the beam.
