"""
Optimizers, used by 'spinn.train' (argument 'method').

An optimizer minimizes the loss for a given number of iterations, starting
from the given weights. 'train' calls it once per weight update of the
balancer (see 'spinn.balance'), with the loss weights fixed during the run.

Own optimizers subclass 'optimizer' and define '__call__'; options are
class attributes, which can be changed per instance:

    class mine(optimizer):
        step = 1e-3
        def __call__(self, loss, grad, w, n, callback=None):
            for i in range(n):
                w = w - self.step*grad(w)
            return OptimizeResult(x=w, fun=loss(w), nit=n, nfev=n,
                                  status=1, success=False)

'train' deep-copies the optimizer, so an optimizer can keep its state
(e.g. moments) in its attributes between the runs of one training.
"""
import numpy as np
import scipy.optimize
from scipy.optimize import OptimizeResult

__all__ = ['optimizer', 'minimize', 'get', 'OptimizeResult']


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

    def __call__(self, loss, grad, w, n, callback=None):
        """Minimize 'loss(w)' (jax functions 'loss' and its gradient 'grad'
        of the flat network weights) for at most 'n' iterations, starting
        from 'w'. Return a scipy OptimizeResult with at least 'x', 'fun',
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

    def __call__(self, loss, grad, w, n, callback=None):
        opts = dict(self.options or {}, maxiter=n)
        if self.method.upper() == 'L-BFGS-B':
            opts.setdefault('maxfun', 10*n)
            opts.setdefault('maxls', 50)
        return scipy.optimize.minimize(
            _f64(loss), np.asarray(w, dtype=np.float64), jac=_f64(grad),
            method=self.method, tol=self.tol, options=opts,
            callback=callback, **(self.kwargs or {}))


def get(m):
    """Optimizer instance from scipy method name (gives 'minimize'), class
    or instance."""
    if isinstance(m, str):
        return minimize(method=m)
    if isinstance(m, type) and issubclass(m, optimizer):
        return m()
    if isinstance(m, optimizer):
        return m
    raise TypeError("optimizer must be a scipy method name or subclass "
                    "spinn.optimize.optimizer")
