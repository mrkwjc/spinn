import importlib
import itertools
import jax
import jax.numpy as jnp


def prod(s1, s2):
    return list(itertools.product(s1, s2))


def acc(s):
    return list(itertools.accumulate(s))


def pairwise(s):
    return list(itertools.pairwise(s))


def pacc(s):
    return list(itertools.pairwise(itertools.accumulate(s)))


def to01c(x):
    assert len(x.shape) == 2
    assert x.shape[0] >= 2

    def _to01(lim):
        a, b = lim.T
        c0 = 1./(b-a)
        c1 = -a * c0
        return c0, c1

    xlim = jnp.asarray([(i.min(), i.max()) for i in x.T])
    ax, bx = _to01(xlim)
    return ax, bx


def from01c(x):
    def _from01(lim):
        c, d = lim.T
        c0 = d - c
        c1 = c
        return c0, c1

    if isinstance(x, int):
        xlim = jnp.asarray([(0., 1.)]*x)
    else:
        xlim = jnp.asarray([(i.min(), i.max()) for i in x.T])

    ax, bx = _from01(xlim)
    return ax, bx


def actname(f):
    """
    Name stored for the activation 'f': a string is kept, a function from
    jax.nn or jax.numpy gets its name there ('tanh', 'gelu'), a module-level
    function its dotted path ('myacts.swish2'). Lambdas and local functions
    are kept as they are (usable, but not savable).
    """
    if isinstance(f, str):
        actfunc(f)                      # unknown names fail here, early
        return f
    name = getattr(f, '__name__', '')
    for mod in (jax.nn, jax.numpy):
        if getattr(mod, name, None) is f:
            return name
    qual = getattr(f, '__qualname__', '')
    if qual and '<' not in qual:        # not '<lambda>', not '<locals>'
        return '%s.%s' % (f.__module__, qual)
    return f


def actfunc(f):
    """Function for the activation 'f' (name or function)."""
    if callable(f):
        return f
    if f == 'identity':
        return lambda x: x
    if f == 'linear':
        return lambda x, a=1., b=0.: a*x + b
    for mod in (jax.nn, jax.numpy):
        g = getattr(mod, f, None)
        if callable(g):
            return g
    if '.' in f:
        module, name = f.rsplit('.', 1)
        return getattr(importlib.import_module(module), name)
    raise ValueError("unknown activation '%s' (use a name from jax.nn or "
                     "jax.numpy, or 'module.function')" % f)


def unroll_fp(f, p, arch):
    nl = len(arch)
    if f is None:
        f = ['identity'] + ['softplus']*(nl-2) + ['identity']
    elif isinstance(f, str) or callable(f):
        f = ['identity'] + [f]*(nl-2) + ['identity']
    elif isinstance(f, list):
        if len(f) == 3:
            f = [f[0]] + [f[1]]*(nl-2) + [f[2]]
        elif len(f) == len(arch):
            f = f
        else:
            raise ValueError(f"{len(f)} funcs given for {len(arch)} layers.")
    else:
        raise ValueError("'f' have to be None, str, callable or a list.")
    f = [actname(fi) for fi in f]
    if p is None:
        p = [()]*nl
    elif isinstance(p, tuple):
        p = [()] + [p]*(nl-2) + [()]
    elif isinstance(p, list):
        if len(p) == 3:
            p = [p[0]] + [p[1]]*(nl-2) + [p[2]]
        elif len(p) == len(arch):
            p = p
        else:
            raise ValueError(f"{len(p)} params given for {len(arch)} layers")
    else:
        raise ValueError("'p' have to be None, tuple or a list.")
    return f, p


def hcube(dims, n=11, origin=None):
    dims = (1.,)*dims if isinstance(dims, int) else dims
    dims = (dims,) if isinstance(dims, float) else dims
    d = len(dims)
    n = (n,)*d if isinstance(n, int) else n
    origin = (0,)*d if origin is None else origin
    xyz = [jnp.linspace(origin[i], dims[i]+origin[i], n[i]) for i in range(d)]
    XYZ = jnp.meshgrid(*xyz, indexing='ij')
    return jnp.vstack(tuple(map(jnp.ravel, XYZ))).T


def left(pts):
    x = pts[:, 0]
    return pts[jnp.isclose(x, min(x))]


def right(pts):
    x = pts[:, 0]
    return pts[jnp.isclose(x, max(x))]


def bottom(pts):
    x = pts[:, 1]
    return pts[jnp.isclose(x, min(x))]


def top(pts):
    x = pts[:, 1]
    return pts[jnp.isclose(x, max(x))]
