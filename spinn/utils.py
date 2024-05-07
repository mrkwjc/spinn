import itertools
import functools
import random
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


def meshgrid(*arrs):
    arrs = tuple(reversed(arrs))
    lens = tuple(map(len, arrs))
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s
    ans = []
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = jnp.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    return tuple(ans)


def hcube(dims, n=11, origin=None):
    dims = (1.,)*dims if isinstance(dims, int) else dims
    dims = (dims,) if isinstance(dims, float) else dims
    d = len(dims)
    n = (n,)*d if isinstance(n, int) else n
    origin = (0,)*d if origin is None else origin
    xyz = [jnp.linspace(origin[i], dims[i]+origin[i], n[i]) for i in range(d)]
    XYZ = meshgrid(*xyz)
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


def uniform(low=0., high=1., shape=(), key=None):
    shape = (shape,) if isinstance(shape, int) else shape
    key = random.randint(0, 10000000000) if key is None else key
    key = jax.random.PRNGKey(key) if isinstance(key, int) else key
    return jax.random.uniform(key, shape=shape, minval=low, maxval=high)


def randomweights(arch, key=None):
    W = []
    wshp = list(itertools.pairwise(arch))
    bshp = arch[1:]
    for i, n in enumerate(arch[:-1]):
        bound = 2.38/jnp.sqrt(n+1)
        w = uniform(-bound, bound, shape=wshp[i])
        b = uniform(-bound, bound, shape=bshp[i])
        W += [[w, b]]
    return W


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0, None))
def simpleprop(w, x, c):
    ax, bx, ay, by = c
    y = ax*x + bx
    for wi, bi in w[:-1]:
        yi = y.dot(wi) + bi
        y = jax.nn.softplus(yi)
    wi, bi = w[-1]
    y = y.dot(wi) + bi
    y = ay*y + by
    return y
