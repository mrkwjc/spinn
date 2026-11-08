"""
Balancing of loss terms, used by 'spinn.train' (argument 'balance').

A balancer returns the weights 'lmbd' of the loss terms, sum(lmbd * terms).
'train' runs the optimizer for 'inner' iterations with fixed weights, then
asks the balancer for new ones, and repeats this until the weights settle
(the optimizer converged and no weight changed by more than 10%), after
'outer' updates or when 'maxiter' is used up. (When the optimizer has
converged, further updates hardly change the weights and only restart it,
so a stricter stop costs time without gaining accuracy.)

Own rules subclass 'balancer' and define '__call__'; options are class
attributes, which can be changed per instance:

    class mine(balancer):
        p = 2.
        def __call__(self, model, w, *args):
            T = model._terms(w, *args)   # values of the terms (jax array)
            return ...                   # one weight per term

    B.train(*args, balance=mine(p=3., inner=500))

In 'train' a balancer is given by name ('grad'), class (grad), instance
(grad(lmax=1e3)), as a sequence of fixed weights or None (equal weights).
"""
import numpy as np


class balancer:
    """
    Base class of balancers. Options:

        inner - optimizer iterations between weight updates (default 500)
        outer - maximum number of weight updates (default unlimited)
    """
    inner = 500
    outer = np.inf

    def __init__(self, **options):
        for k, v in options.items():
            if not hasattr(self, k):
                raise TypeError("%s has no option '%s'"
                                % (type(self).__name__, k))
            setattr(self, k, v)

    def __call__(self, model, w, *args):
        """Weights of the loss terms (one per term)."""
        raise NotImplementedError

    def __repr__(self):
        opts = {k: getattr(self, k) for k in dir(self)
                if not k.startswith('_') and not callable(getattr(self, k))}
        return '%s(%s)' % (type(self).__name__,
                           ', '.join('%s=%r' % kv for kv in opts.items()))


def onesided(v, lmax):
    """
    One-sided weights from term measures 'v' (values, gradient norms):
    lmbd[i] = max(1, v[0]/v[i]), capped at 'lmax'. Terms are only weighted
    up relative to the first (reference) term; two-sided weights fail when
    a term is nearly satisfied at some point (e.g. a contact condition
    before contact): its tiny measure gives it a huge weight and the other
    terms are neglected.
    """
    v = np.asarray(v, np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        lm = np.where(v > 0, v[0]/v, lmax)
    lm = np.clip(np.nan_to_num(lm, nan=1.), 1., lmax)
    lm[0] = 1.
    return lm


class grad(balancer):
    """
    lmbd[i] = max(1, |grad terms[0]|/|grad terms[i]|), so that no term
    pulls the network weaker than the reference (Wang et al. 2021,
    "learning rate annealing", here between optimizer runs). Option
    'lmax' - maximum weight (default 1e6).
    """
    lmax = 1e6

    def __call__(self, model, w, *args):
        return onesided(model._termsgn(w, *args), self.lmax)


class fixed(balancer):
    """Fixed weights 'lmbd' (equal if None), single optimizer run."""
    inner = np.inf
    outer = 1

    def __init__(self, lmbd=None, **options):
        super().__init__(**options)
        self.lmbd = lmbd

    def __call__(self, model, w, *args):
        n = len(model._terms(w, *args))
        if self.lmbd is None:
            return np.ones(n)
        lmbd = np.ravel(np.asarray(self.lmbd, np.float64))
        if len(lmbd) != n:
            raise ValueError("%d weights given for %d terms"
                             % (len(lmbd), n))
        return lmbd


def get(b):
    """Balancer instance from name, class, instance, weights or None."""
    if b is None:
        return fixed()
    if isinstance(b, str):
        cls = globals().get(b)
        if not (isinstance(cls, type) and issubclass(cls, balancer)
                and cls is not balancer):
            names = [k for k, c in globals().items() if isinstance(c, type)
                     and issubclass(c, balancer) and c is not balancer]
            raise ValueError("unknown balancer '%s', choose from %s"
                             % (b, names))
        return cls()
    if isinstance(b, type) and issubclass(b, balancer):
        return b()
    if isinstance(b, balancer):
        return b
    if callable(b):
        raise TypeError("balancing rules must subclass "
                        "spinn.balance.balancer")
    return fixed(b)
