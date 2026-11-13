"""
Balancing of loss terms, used by 'spinn.train' (argument 'balance').

A balancer returns the weights of the loss terms, the lambdas 'lmbd' in
sum(lmbd * terms) ('weights' alone are the weights of the network).
'train' runs the optimizer for 'inner' iterations (500 for lbfgs, 2000 for
adam, see 'spinn.optimize') with fixed lambdas, then asks the balancer for
new ones, and repeats this until the lambdas settle (the optimizer
converged and no lambda changed by more than 10%), two optimizer runs in a
row make less than 10 iterations (nothing left to gain), after 'outer'
updates or when 'maxiter' is used up. (When the optimizer has converged,
further updates hardly change the lambdas and only restart it, so a
stricter stop costs time without gaining accuracy.)

Own rules subclass 'balancer' and define '__call__'; options are class
attributes, which can be changed per instance:

    class mine(balancer):
        p = 2.
        def __call__(self, model, w, lmbd, *args):
            T = model._terms(w, *args)   # values of the terms (jax array)
            return ...                   # one lambda per term

'w' are the network weights, 'lmbd' the lambdas used in the loss so far
(None at the first call), 'args' the arguments of 'terms'.

    B.train(*args, balance=mine(p=3., inner=500))

'train' works on a copy of the balancer, so a balancer may keep a state
between updates (every training starts afresh). A balancer with
several phases moves to the next one in 'advance': when 'train' would
stop by the rules above (lambdas settled, short runs) it calls 'advance'
and goes on if it returns True. With an 'anytime' optimizer (e.g. adam,
whose runs do not converge) settled lambdas are enough to advance, and the
last phase runs until 'maxiter'.

In 'train' a balancer is given by name or expression (see below), class
(curv), instance (curv(lmax=1e3)), as a sequence of fixed lambdas or None
(equal lambdas).

Rules: 'curv' (the default), 'grad' and 'sens'. They combine into
expressions written as in Python: '*' multiplies lambdas, '**' raises them
to a number, and a comma separates phases (the next rule takes over when
'train' would stop with the previous one):

    'curv**.5*sens'       - sqrt(curv)*sens from the start
    'curv, curv**.5*sens' - curv, then sqrt(curv)*sens (most accurate so
                            far, typically at 1.1-1.6 times the cost of
                            curv); shortcut 'curvsens'
    'curv, sens'          - curv, then pure sensitivity lambdas

Each name in an expression is a new instance with default options; with
other options build the balancers: phases(curv(), product((curv(), .5),
(sens(maxcg=100), 1))).
"""
import ast
import numpy as np
import jax
import jax.numpy as jnp


# Derivatives of the model used by the balancers (jitted; the model is a
# pytree argument)

@jax.jit
def _termsvhv(model, w, vs, *args):
    """
    Curvatures v.H.v of each loss term along the rows v of 'vs' (H -
    Hessian of the term w.r.t. weights), exact second directional
    derivatives without forming H; shape (len(vs), number of terms).
    """
    def T(w):
        return model._terms(w, *args)

    def vhv(v):
        return jax.jvp(lambda u: jax.jvp(T, (u,), (v,))[1], (w,), (v,))[1]
    return jax.vmap(vhv)(vs)


@jax.jit
def _termsjac(model, w, *args):
    """Gradients of each loss term w.r.t. weights (rows)."""
    return jax.jacrev(lambda w: model._terms(w, *args))(w)


@jax.jit
def _losshv(model, w, lmbd, vs, *args):
    """Products H.v of the Hessian of the weighted loss with rows v."""
    def g(u):
        return model._lossg(u, lmbd, *args)
    return jax.vmap(lambda v: jax.jvp(g, (w,), (v,))[1])(vs)


@jax.jit
def _sol(model, w, x):
    """
    Solution (network outputs) at the points x, flattened: x is an array
    (all outputs at all points) or a tuple with one array (or None) per
    output (output k only at x[k], e.g. piecewise models).
    """
    model.w = w
    if isinstance(x, (tuple, list)):
        if len(x) != model.no:
            raise ValueError("domain tuple has %d entries for %d outputs"
                             % (len(x), model.no))
        return jnp.concatenate([model(xk)[:, k].ravel()
                                for k, xk in enumerate(x) if xk is not None])
    return model(x).ravel()


def _solfields(model, x):
    """Output index of each entry of '_sol(model, w, x)'."""
    if isinstance(x, (tuple, list)):
        return np.concatenate([np.full(len(xk), k)
                               for k, xk in enumerate(x) if xk is not None])
    return np.tile(np.arange(model.no), len(x))


@jax.jit
def _soljvp(model, w, x, vs):
    """Changes (du/dw).v of the solution at x along the rows v of vs."""
    return jax.vmap(lambda v: jax.jvp(lambda u: _sol(model, u, x),
                                      (w,), (v,))[1])(vs)


def _like(model, a):
    """'a' as a jax array of the dtype of the network weights."""
    return jnp.asarray(a, jnp.asarray(model.w).dtype)


def _ratios(v, r, zero, lmax):
    """
    Lambdas v[r]/v[i] from term measures 'v' with term r as the reference
    (lambda 1); terms with v[i] = 0 get 'zero'. Lambdas are kept between
    1 and 'lmax'.
    """
    v = np.asarray(v, np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        lm = np.where(v > 0, v[r]/v, zero)
    lm = np.clip(np.nan_to_num(lm, nan=1.), 1., lmax)
    lm[r] = 1.
    return lm


class balancer:
    """
    Base class of balancers. Options:

        inner - optimizer iterations between lambda updates (default None:
                the optimizer's, 500 for lbfgs and minimize, 2000 for adam)
        outer - maximum number of lambda updates (default unlimited)
    """
    inner = None
    outer = np.inf

    def __init__(self, **options):
        for k, v in options.items():
            if not hasattr(self, k):
                raise TypeError("%s has no option '%s'"
                                % (type(self).__name__, k))
            setattr(self, k, v)

    def __call__(self, model, w, lmbd, *args):
        """Lambdas, the weights of the loss terms (one per term), from the
        network weights 'w', the lambdas so far 'lmbd' (None at the start)
        and the arguments 'args' of 'terms'."""
        raise NotImplementedError

    def advance(self):
        """Move to the next phase; False if there is none. 'train' calls it
        until it returns False, then no more."""
        return False

    def __repr__(self):
        opts = {k: getattr(self, k) for k in dir(self)
                if not k.startswith('_') and not callable(getattr(self, k))}
        return '%s(%s)' % (type(self).__name__,
                           ', '.join('%s=%r' % kv for kv in opts.items()))


def onesided(v, lmax):
    """
    One-sided lambdas from term measures 'v' (values, gradient norms):
    lmbd[i] = max(1, v[0]/v[i]), capped at 'lmax'. Terms are only weighted
    up relative to the first (reference) term; two-sided lambdas fail when
    a term is nearly satisfied at some point (e.g. a contact condition
    before contact): its tiny measure gives it a huge lambda and the other
    terms are neglected.
    """
    return _ratios(v, 0, lmax, lmax)


class grad(balancer):
    """
    lmbd[i] = max(1, |grad terms[0]|/|grad terms[i]|), so that no term
    pulls the network weaker than the reference (Wang et al. 2021,
    "learning rate annealing", here between optimizer runs). Option
    'lmax' - maximum lambda (default 1e6).
    """
    lmax = 1e6

    def __call__(self, model, w, lmbd, *args):
        G = _termsjac(model, _like(model, w), *args)
        return onesided(np.linalg.norm(np.asarray(G, np.float64), axis=1),
                        self.lmax)


class curv(balancer):
    """
    Equal curvature of the weighted terms: the stiffest term is the
    reference and the others are weighted up to its curvature,

        lmbd[i] = h[r]/h[i],   r = argmax(h),

    where h[i] = |trace of the Hessian of terms[i]| w.r.t. the network
    weights, estimated from 'probes' random +-1 vectors v (Hutchinson,
    v.H.v as second directional derivatives, no Hessian is formed).
    Lambdas are capped at 'lmax'; the cap is high, as terms of fields of
    very different size (multiphysics) may need lambdas of 1e10 and more. A
    term with zero curvature (independent of the network weights, e.g. at a
    point not reached by the network) gets lambda 1. Options: 'probes'
    (default 4), 'lmax' (default 1e12), 'seed' of the probes (default 0).
    """
    lmax = 1e12
    probes = 4
    seed = 0

    def __call__(self, model, w, lmbd, *args):
        rng = np.random.default_rng(self.seed)
        vs = rng.choice([-1., 1.], size=(self.probes, np.size(w)))
        h = _termsvhv(model, _like(model, w), _like(model, vs), *args)
        h = np.abs(np.mean(np.asarray(h, np.float64), axis=0))
        return _ratios(h, int(np.argmax(h)), 1., self.lmax)


def cg(hv, B, maxcg=50, tol=1e-6):
    """
    Solves A x = b for the rows b of B by conjugate gradients, all rows at
    once; hv(V) returns the products A v for the rows v of V. A row stops
    at relative residual 'tol', after 'maxcg' iterations or at negative
    curvature (v.A.v <= 0, then the solution so far is kept, as in
    truncated Newton methods). Returns the solutions (rows).
    """
    B = np.asarray(B, np.float64)
    X = np.zeros_like(B)
    R = B.copy()
    P = R.copy()
    rr = np.sum(R*R, 1)
    stop = np.sqrt(rr) <= tol*np.linalg.norm(B, axis=1)
    for _ in range(maxcg):
        act = ~stop
        if not act.any():
            break
        AP = hv(P)
        pAp = np.sum(P*AP, 1)
        neg = act & (pAp <= 0)
        stop |= neg
        act &= ~neg
        a = np.where(act, rr/np.where(pAp > 0, pAp, 1.), 0.)
        X += a[:, None]*P
        R -= a[:, None]*AP
        rr2 = np.sum(R*R, 1)
        stop |= act & (np.sqrt(rr2) <= tol*np.linalg.norm(B, axis=1))
        beta = np.where(act, rr2/np.maximum(rr, 1e-300), 0.)
        P = np.where(act[:, None], R + beta[:, None]*P, P)
        rr = np.where(act, rr2, rr)
    return X


class sens(balancer):
    """
    Lambdas from the sensitivity of the solution to each term:

        dw[i] = -H^-1 lmbd[i] grad terms[i],    du[i] = (du/dw) dw[i]
        s[i]  = |du[i]| / sqrt(terms[i]),       lambdas ~ s

    dw[i] is the change of the network weights which would remove the
    residual of term i (Newton step of the weighted loss sum(lmbd*terms)),
    du[i] the resulting change of the solution at the points args[0]; s[i]
    is the solution change per unit residual. H is damped by damp*(its
    largest eigenvalue) and the systems are solved by 'maxcg' conjugate
    gradient iterations without forming H. 'lmbd' are the lambdas used in
    the loss so far (equal if None).

    args[0] is the domain: an array (all outputs at all points) or a tuple
    with one array (or None) per output (output k only at args[0][k], e.g.
    piecewise models). With several outputs their changes are scaled by
    the RMS of each output (scale='rms') or not ('none'). log(s) is
    averaged over the updates with avg='ema' (weight 1/2 to the new value)
    or not ('none'). The smallest lambda is 1, lambdas are capped at
    'lmax'. A term at exactly zero has no sensitivity (zero gradient) and
    keeps its previous lambda.

    The terms most able to move the solution get the largest lambdas: e.g.
    Dirichlet conditions, which set the solution, above Neumann ones. The
    sensitivities are meaningful for a network already trained, and in
    physical units the Hessian of plainly weighted terms is beyond float32;
    so use it after 'curv' and combined with it: 'curv, curv**.5*sens'.
    Options: 'lmax' (default 1e12), 'avg' ('ema'), 'damp' (1e-6), 'maxcg'
    (50), 'scale' ('rms'), 'seed' (0).
    """
    lmax = 1e12
    avg = 'ema'
    damp = 1e-6
    maxcg = 50
    scale = 'rms'
    seed = 0
    _lf = None

    def __call__(self, model, w, lmbd, *args):
        wj = _like(model, w)
        T = np.asarray(model._terms(wj, *args), np.float64)
        lm = np.ones(len(T)) if lmbd is None else np.asarray(lmbd, np.float64)
        lmj = _like(model, lm)
        G = np.asarray(_termsjac(model, wj, *args), np.float64)

        def hv(V):
            V = _like(model, V)
            return np.asarray(_losshv(model, wj, lmj, V, *args), np.float64)
        # largest eigenvalue of H (power iterations) sets the damping
        v = np.random.default_rng(self.seed).standard_normal((1, wj.size))
        hmax = 0.
        for _ in range(30):
            u = hv(v)
            hmax = np.linalg.norm(u)
            if not hmax > 0:
                break
            v = u/hmax
        mu = self.damp*hmax
        dW = cg(lambda V: hv(V) + mu*V, -lm[:, None]*G, self.maxcg)
        x = args[0]
        du = np.asarray(_soljvp(model, wj, x, _like(model, dW)), np.float64)
        if self.scale == 'rms':
            fid = _solfields(model, x)
            y = np.asarray(_sol(model, wj, x), np.float64)
            sig = np.array([np.sqrt(np.mean(y[fid == k]**2))
                            if np.any(fid == k) else 1.
                            for k in range(model.no)])
            sig = np.maximum(sig, 1e-12*max(sig.max(), 1e-300))
            du = du/sig[fid]
        elif self.scale != 'none':
            raise ValueError("scale must be 'rms' or 'none'")
        # a term at exactly 0 (reached in float32) has a zero gradient, so
        # its sensitivity is 0/0: it keeps its previous value (the median
        # of the others at the first update) and is not the reference
        ok = T > 0
        if not ok.any():
            lf = np.zeros(len(T)) if self._lf is None else self._lf
            self._lf = lf
            return np.clip(np.exp(lf), 1., self.lmax)
        s = np.linalg.norm(du, axis=1)/np.sqrt(np.where(ok, T, 1.))
        lf = np.log(np.maximum(s, 1e-300))
        lf -= lf[ok].min()
        if not ok.all():
            lf[~ok] = (np.median(lf[ok]) if self._lf is None
                       else self._lf[~ok])
        lf -= lf.min()
        if self.avg == 'ema' and self._lf is not None:
            lf = 0.5*lf + 0.5*self._lf
        elif self.avg not in ('ema', 'none'):
            raise ValueError("avg must be 'ema' or 'none'")
        self._lf = lf
        return np.clip(np.exp(lf), 1., self.lmax)


class product(balancer):
    """
    Product of powers of the lambdas of other balancers (parts: pairs
    (balancer, a) or balancers, a = 1),

        lmbd = prod(b(...)**a for b, a in parts),

    scaled so that the smallest is 1 and capped at 'lmax'.
    """
    lmax = 1e12

    def __init__(self, *parts, **options):
        super().__init__(**options)
        self.parts = tuple(p if isinstance(p, tuple) else (p, 1.)
                           for p in parts)

    def __call__(self, model, w, lmbd, *args):
        lm = 1.
        for b, a in self.parts:
            lm = lm*np.asarray(b(model, w, lmbd, *args), np.float64)**a
        return np.clip(lm/np.min(lm), 1., self.lmax)


class phases(balancer):
    """
    Balancers used one after another: the next one takes over when 'train'
    would stop with the current one (see 'advance').
    """
    def __init__(self, *phases, **options):
        super().__init__(**options)
        self.phases = phases
        self._k = 0

    def advance(self):
        if self._k + 1 < len(self.phases):
            self._k += 1
            return True
        return False

    def __call__(self, model, w, lmbd, *args):
        return self.phases[self._k](model, w, lmbd, *args)


class fixed(balancer):
    """Fixed lambdas 'lmbd' (equal if None), single optimizer run."""
    inner = np.inf
    outer = 1

    def __init__(self, lmbd=None, **options):
        super().__init__(**options)
        self.lmbd = lmbd

    def __call__(self, model, w, lmbd, *args):
        n = len(model._terms(w, *args))
        if self.lmbd is None:
            return np.ones(n)
        lm = np.ravel(np.asarray(self.lmbd, np.float64))
        if len(lm) != n:
            raise ValueError("%d lambdas given for %d terms" % (len(lm), n))
        return lm


RULES = ('curv', 'grad', 'sens')    # names usable in expressions
ALIASES = {'curvsens': 'curv, curv**.5*sens'}   # shortcuts (whole string)


def parse(expr):
    """
    Balancer from an expression: names of RULES combined by '*' (product),
    '**' (power, a number) and ',' (phases), e.g. 'curv, curv**.5*sens', or
    a shortcut from ALIASES.
    """
    expr = ALIASES.get(expr.strip(), expr)

    def fail(msg):
        raise ValueError("%s in balancer '%s'; use names %s with '*', "
                         "'**' and ',' (phases), e.g. 'curv, curv**.5*sens'"
                         % (msg, expr, list(RULES)))

    def number(n):
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub,
                                                            ast.UAdd)):
            v = number(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if (isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)):
            return float(n.value)
        fail("exponent must be a number")

    def factors(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            return factors(n.left) + factors(n.right)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            return [(b, a*number(n.right)) for b, a in factors(n.left)]
        if isinstance(n, ast.Name):
            if n.id not in RULES:
                fail("unknown name '%s'" % n.id)
            return [(globals()[n.id](), 1.)]
        fail("unsupported syntax")

    def rule(n):
        f = factors(n)
        return f[0][0] if len(f) == 1 and f[0][1] == 1. else product(*f)
    try:
        tree = ast.parse(expr.strip(), mode='eval').body
    except SyntaxError:
        fail("syntax error")
    if isinstance(tree, ast.Tuple):
        if len(tree.elts) < 2:
            fail("unsupported syntax")
        return phases(*[rule(n) for n in tree.elts])
    return rule(tree)


def get(b):
    """Balancer instance from name or expression, class, instance, lambdas
    or None."""
    if b is None:
        return fixed()
    if isinstance(b, str):
        return parse(b)
    if isinstance(b, type) and issubclass(b, balancer):
        return b()
    if isinstance(b, balancer):
        return b
    if callable(b):
        raise TypeError("balancing rules must subclass "
                        "spinn.balance.balancer")
    return fixed(b)
