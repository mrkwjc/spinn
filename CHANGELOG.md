# Changelog

## Unreleased

### Changed (may need changes in your code)

- **The default optimizer is `'lbfgs'`**: L-BFGS of optax, run in a
  compiled loop on the device of the network (no copies to the host between
  iterations). On the examples it is 1.5–2.7× faster than scipy's L-BFGS-B
  on the GPU, with equal or better accuracy. `method='L-BFGS-B'` gives the
  previous behaviour. scipy's arguments (`options`, `bounds`, ...) need a
  scipy method; with `'lbfgs'` they raise a `TypeError`. `callback` is
  called every 100 iterations (option `chunk`) with the best network weights
  so far.
- **optax** is a new dependency.
- **Wording:** the weights of the loss terms (`lmbd`) are called *lambdas*
  in the documentation; *weights* are those of the network.

### Added

- **Optimizer classes** in the new module `spinn.optimize`, like the
  balancers: `lbfgs` (options `tol`, `patience`, `memory_size`, `chunk`),
  `adam` (Adam of optax on the device; learning rate `lr`, a number or a
  schedule over the whole training, e.g. `adam(lr=1e-2)`; its runs do not
  converge, so balancers with phases, e.g. `'curvsens'`, move on once the
  lambdas settle: option `anytime` of the optimizers; by default it runs
  2000 iterations between lambda updates and 30000 in all),
  `minimize` (any scipy method, e.g. `minimize(method='BFGS', tol=1e-8)`),
  and your own optimizer as a subclass of `spinn.optimize.optimizer`.
  `method` accepts a name, a class or an instance. The optimizer sets the
  defaults of `maxiter` and of the balancer's `inner` (`balancer.inner` is
  now `None`, meaning the optimizer's).
- **Best of several initial network weights:** `train(..., seeds=3)` (or a
  list of seeds) trains from each and keeps the training with the lowest
  loss; `res.seed`, `res.seeds` and `res.scores` report them.

### Fixed

- `'sens'` lambdas when a term is exactly zero (in float32): the other
  lambdas went to the cap, 1e12, and training failed (`'curvsens'` on the
  cantilever example).

## 0.3.0 (2026-11-11)

The problem is now defined by its loss terms, and `train` balances them
automatically.

### Changed (may need changes in your code)

- **`terms` replaces `loss`.** Define `terms` to return the loss terms, with
  the equation (PDE residual) first and the conditions after it. `train`
  minimizes their weighted sum. A `loss` method still works: it is treated
  as a single term and minimized as before, with a `FutureWarning`. The
  `loss`, `lossg` and `lossvg` methods of `spinn` are removed.
- **`eqc` and `ineqc` constraints are removed** (they switched training to
  SLSQP). Defining them raises an error; return the constraints as terms
  instead.
- **`hcube` returns the points in the standard `'ij'` order**: the last
  coordinate varies fastest, so `x.reshape(n1, n2, d)[i, j]` is the point
  (x_i, y_j). Scripts that reshape grids for indexing need checking; plots
  with `contourf` or `plot_surface` keep working.
- **Initial weights for a given `key` change.** `initw` drew every layer
  from the same random numbers; all layers now come from one generator,
  and NumPy's global random state is left alone.
- **`tol` defaults to 1e-10** (was 1e-5), and `maxiter` limits the total
  number of iterations (default 15000).
- **`disp`** prints spinn's progress after each optimizer run and is no
  longer passed to scipy (use `options={'disp': True}` for scipy's output).
- **Python 3.10 or newer** is required.

### Added

- **Automatic balancing of the terms**, `train(..., balance=...)`, in the
  new module `spinn.balance`:
  - `'curv'` (default): weights to equal curvature of the terms, with the
    stiffest term as the reference;
  - `'grad'`: weights to equal gradient norms (learning rate annealing);
  - `'sens'`: weights from the sensitivity of the solution to each term;
  - expressions combining them, e.g. `'curv, curv**.5*sens'` (curv, then
    corrected by sensitivities; shortcut `'curvsens'`, the most accurate on
    the examples);
  - fixed weights (`balance=[1., 10., 10.]`), `None` for a plain sum, or
    your own rule as a subclass of `spinn.balance.balancer`.
- **`method`** chooses the scipy optimizer (default `'L-BFGS-B'`).
- **Results:** `res.lmbd` (the final weights), `res.terms` (the unweighted
  terms) and `res.nouter` (the number of optimizer runs).
- **Activations by name:** `f` accepts any function name from `jax.nn` or
  `jax.numpy` (`'tanh'`, `'sin'`, `'gelu'`, ...), a function (stored by its
  name) or `'module.function'`. `dump` saves the names, so `load` restores
  the activations.
- **`spinn.__version__`.**
- Docstrings for the `spinn` class and its main methods, and a usage section
  in the README.
- Examples: `heat`, `heat_plate`, `beam_support` and `cantilever` (with a
  tip support), all compared with exact solutions; `beam` with the bending
  stiffness `EI`.

### Fixed

- Values passed to scipy are float64 (scipy 1.16 requires this for some
  optimizers).
- `dump` failed for activations with plain-number parameters, e.g.
  `f='elu', p=(0.5,)`.
- L-BFGS-B uses `maxls=50` and a higher `maxfun`, so long or stiff
  trainings no longer stop early.

## 0.2.0 (2026-09-27)

- L-BFGS-B is the default optimizer.
- `'sin'` activation.
- `dump` and `load` save networks as plain lists; networks can be pickled.
- Corrected `meshgrid`.

## 0.1.0 (2024-05-07)

First release.
