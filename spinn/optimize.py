"""
Optimizers, used by 'spinn.train' (argument 'method').

An optimizer minimizes the loss for a given number of iterations, starting
from the given weights. 'train' calls it once per weight update of the
balancer (see 'spinn.balance'), with the loss weights fixed during the run.

Own optimizers subclass 'optimizer' and define '__call__'; options are
class attributes, which can be changed per instance:

    class mine(optimizer):
        step = 1e-3
        def __call__(self, model, w, lmbd, *args, n, callback=None):
            for i in range(n):
                w = w - self.step*model._lossg(w, lmbd, *args)
            return OptimizeResult(x=w, fun=model._loss(w, lmbd, *args),
                                  nit=n, nfev=n, status=1, success=False)

'model._loss(w, lmbd, *args)' is the loss, sum(lmbd * terms), and
'model._lossg' its gradient; both are jitted and can be used inside own
jitted code (the model is a pytree), so the whole run can be compiled once
per problem.

'train' deep-copies the optimizer, so an optimizer can keep its state
(e.g. moments) in its attributes between the runs of one training.
"""
import inspect
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.optimize
from optax import tree_utils as otu
from scipy.optimize import OptimizeResult

__all__ = ['optimizer', 'minimize', 'lbfgs', 'get', 'OptimizeResult']


class optimizer:
    """
    Base class of optimizers. Options:

        tol - tolerance of the stop criteria (meaning depends on the
              optimizer)
    """
    tol = None

    def __init__(self, **options):
        for k, v in options.items():
            if not hasattr(self, k):
                raise TypeError("%s has no option '%s'"
                                % (type(self).__name__, k))
            setattr(self, k, v)

    def __call__(self, model, w, lmbd, *args, n, callback=None):
        """Minimize the loss of 'model', model._loss(w, lmbd, *args), over
        the flat network weights 'w' for at most 'n' iterations, starting
        from 'w' (numpy float64); 'lmbd' (weights of the terms, in the
        precision of the network) and 'args' (arguments of 'terms') are
        fixed. Return a scipy OptimizeResult with at least 'x', 'fun',
        'nit', 'nfev', 'status' (99 if stopped by 'callback') and
        'success'."""
        raise NotImplementedError

    def __repr__(self):
        opts = {k: getattr(self, k) for k in dir(self)
                if not k.startswith('_') and not callable(getattr(self, k))}
        return '%s(%s)' % (type(self).__name__,
                           ', '.join('%s=%r' % kv for kv in opts.items()))


def _f64(fun):
    def wrapped(w):
        return np.asarray(fun(w), dtype=np.float64)
    return wrapped


class minimize(optimizer):
    """
    scipy.optimize.minimize. Options:

        method  - scipy method (default 'L-BFGS-B')
        tol     - tolerance passed to minimize (default 1e-10)
        options - method options passed to minimize; 'maxiter' is set by
                  'train', and for L-BFGS-B 'maxfun' defaults to 10*maxiter
                  and 'maxls' to 50 (scipy's defaults stop long or stiff
                  trainings early)
        kwargs  - other keyword arguments of minimize (e.g. 'bounds')

    Values passed to scipy are float64, the loss is computed in the
    precision of the network.
    """
    method = 'L-BFGS-B'
    tol = 1e-10
    options = None
    kwargs = None

    def __call__(self, model, w, lmbd, *args, n, callback=None):
        def loss(w):
            return model._loss(w, lmbd, *args)

        def grad(w):
            return model._lossg(w, lmbd, *args)
        opts = dict(self.options or {}, maxiter=n)
        if self.method.upper() == 'L-BFGS-B':
            opts.setdefault('maxfun', 10*n)
            opts.setdefault('maxls', 50)
        return scipy.optimize.minimize(
            _f64(loss), np.asarray(w, dtype=np.float64), jac=_f64(grad),
            method=self.method, tol=self.tol, options=opts,
            callback=callback, **(self.kwargs or {}))


def _call(callback, x, fun, nit):
    """Call 'callback' like scipy does: with an OptimizeResult if its
    parameter is named 'intermediate_result', else with the weights."""
    try:
        params = list(inspect.signature(callback).parameters)
    except (TypeError, ValueError):
        params = []
    if params == ['intermediate_result']:
        callback(OptimizeResult(x=x, fun=fun, nit=nit))
    else:
        callback(x)


@partial(jax.jit, static_argnames=('memory_size',))
def _lbfgs_init(w, memory_size):
    return optax.lbfgs(memory_size=memory_size).init(w)


@partial(jax.jit, static_argnames=('memory_size',))
def _lbfgs_run(model, lmbd, args, carry, kmax, tol, patience, memory_size):
    """Steps of optax.lbfgs until iteration 'kmax' or a stop. Carry: weights,
    optax state, iteration, stop reason (0 none, 1 no improvement in
    'patience' steps in a row, 2 small gradient), function evaluations,
    steps without improvement in a row, best weights, best loss."""
    opt = optax.lbfgs(memory_size=memory_size)

    def f(w):
        return model._loss(w, lmbd, *args)
    vg = optax.value_and_grad_from_state(f)

    def cond(c):
        return (c[2] < kmax) & (c[3] == 0)

    def step(c):
        w, st, k, reason, nfev, nst, wb, vb = c
        v, g = vg(w, state=st)
        u, st = opt.update(g, st, w, value=v, grad=g, value_fn=f)
        w = optax.apply_updates(w, u)
        vn = otu.tree_get(st, 'value')
        gn = otu.tree_get(st, 'grad')
        better = vn < vb - tol*jnp.maximum(jnp.abs(vb), 1.)
        wb = jnp.where(better, w, wb)
        vb = jnp.where(better, vn, vb)
        nst = jnp.where(better, 0, nst + 1)
        reason = jnp.where(jnp.max(jnp.abs(gn)) <= tol, 2,
                           jnp.where(nst >= patience, 1, 0))
        nfev = nfev + otu.tree_get(st, 'num_linesearch_steps')
        return w, st, k + 1, reason, nfev, nst, wb, vb
    return jax.lax.while_loop(cond, step, carry)


class lbfgs(optimizer):
    """
    L-BFGS of optax, run in a jitted loop on the device of the network (no
    copies to the host between iterations; in the precision of the network,
    float32 by default). Options:

        tol         - a step improves the loss if it lowers the best loss
                      so far by more than tol*max(|loss|, 1); the run also
                      stops when the largest gradient component is below
                      tol (default 1e-10)
        patience    - the run stops after this many steps in a row without
                      improvement (default 5)
        memory_size - number of stored corrections (default 10)
        chunk       - iterations between calls of 'callback' (default 100)

    The best weights of the run are returned: a step that raises the loss
    (when the line search fails in float32) is never kept.
    """
    tol = 1e-10
    patience = 5
    memory_size = 10
    chunk = 100

    def __call__(self, model, w, lmbd, *args, n, callback=None):
        w = jnp.asarray(w, jnp.asarray(model.w).dtype)
        v = model._loss(w, lmbd, *args)
        z = jnp.asarray(0)
        carry = (w, _lbfgs_init(w, self.memory_size), z, z, z, z, w, v)
        k, status = 0, None
        while status is None:
            kmax = n if callback is None else min(k + self.chunk, n)
            carry = _lbfgs_run(model, lmbd, args, carry, kmax, self.tol,
                               self.patience, self.memory_size)
            k, reason = int(carry[2]), int(carry[3])
            if callback is not None:
                try:
                    _call(callback, np.asarray(carry[6], np.float64),
                          float(carry[7]), k)
                except StopIteration:
                    status = 99
                    break
            if reason:
                status = 0
            elif k >= n:
                status = 1
        message = {0: ('no improvement in %d steps' % self.patience
                       if reason == 1 else 'small gradient'),
                   1: 'maximum number of iterations reached',
                   99: 'stopped by callback'}[status]
        return OptimizeResult(x=np.asarray(carry[6], np.float64),
                              fun=float(carry[7]), nit=k,
                              nfev=int(carry[4]) + 1, status=status,
                              success=status == 0, message=message)


def get(m):
    """Optimizer instance from a name ('lbfgs' gives 'lbfgs', a scipy method
    name 'minimize' with this method), class or instance."""
    if isinstance(m, str):
        if m.lower() == 'lbfgs':
            return lbfgs()
        return minimize(method=m)
    if isinstance(m, type) and issubclass(m, optimizer):
        return m()
    if isinstance(m, optimizer):
        return m
    raise TypeError("optimizer must be a method name or subclass "
                    "spinn.optimize.optimizer")
