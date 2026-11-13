import functools
import itertools
import random
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spinn import utils, graphs, spinn, balance, optimize


# Reference layered network, independent of spinn

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


class TestSpinn:
    def test_simple_layered(self):
        ni, no = 5, 3
        arch = [ni, 30, 40, no]
        x = jnp.asarray(np.random.rand(10, ni))
        t = jnp.asarray(np.random.rand(10, no))
        G = graphs.mlgraph(arch)
        N = spinn(G, initx=x, inity=t)

        # Apply same weights
        w = randomweights(arch)
        ww = [(wi[0].T.ravel(), wi[1]) for wi in w]
        wl, bl = list(zip(*ww))
        ww = jnp.concatenate(wl + bl)
        N.w = ww

        c = utils.to01c(x) + utils.from01c(t)
        y1 = np.array(N(x))
        y2 = np.array(simpleprop(w, x, c))
        print(y1-y2)
        print(y2[0])

        np.testing.assert_almost_equal(y1, y2, 5)

    def test_hcube_order(self):
        # 'ij' order: the last coordinate varies fastest, so
        # x.reshape(n1, n2, n3, 3)[i, j, k] is the point (x_i, y_j, z_k)
        x = utils.hcube((1., 2., 3.), n=(2, 3, 4), origin=(0., 1., 2.))
        assert x.shape == (24, 3)
        g = np.asarray(x).reshape(2, 3, 4, 3)
        np.testing.assert_allclose(g[1, 2, 3], [1., 3., 5.])
        np.testing.assert_allclose(g[0, 1, 2], [0., 2., 4.])
        np.testing.assert_allclose(x[:3], [[0., 1., 2.], [0., 1., 3.],
                                           [0., 1., 4.]])


class fit2(spinn):
    """Data fit split in two terms of very different scale."""
    @jax.jit
    def terms(self, w, x, t):
        self.w = jnp.asarray(w)
        y = self(x)
        return self.mse(y, t), self.mse(y[:2], 1e-3*t[:2])


class single(spinn):
    """Own combination of terms returned as a single value."""
    @jax.jit
    def terms(self, w, x, t):
        self.w = jnp.asarray(w)
        return 2*self.mse(self(x), t)


class poisson(spinn):
    """u'' = -pi**2 sin(pi*x) on [0, 1], u(0) = u(1) = 0; two outputs:
    output 0 lives on [0, 1/2], output 1 on [1/2, 1] (domain tuple)."""
    @jax.jit
    def terms(self, w, x, bx):
        self.w = jnp.asarray(w)
        x1, x2 = x

        def f(x):
            return -jnp.pi**2*jnp.sin(jnp.pi*x).ravel()
        u1xx, u2xx = self.propx([[x1, 0, 0, 0], [x2, 1, 0, 0]])
        (u1,), (u2,) = self.propx([[bx[:1], 0]]), self.propx([[bx[1:], 1]])
        m = jnp.asarray([[.5]])
        um1, um2, du1, du2 = self.propx([[m, 0], [m, 1], [m, 0, 0],
                                         [m, 1, 0]])
        return (self.mse(u1xx, f(x1)) + self.mse(u2xx, f(x2)),
                u1[0]**2 + u2[0]**2,
                (um1[0] - um2[0])**2 + (du1[0] - du2[0])**2)


def data():
    x = jnp.linspace(0., 1., 20)[:, None]
    return x, jnp.sin(3*x)


class TestTrain:
    def test_default_terms(self):
        x, t = data()
        N = spinn([1, 8, 1], f=jnp.tanh, key=1)
        np.testing.assert_allclose(N._terms(N.w, x, t)[0], N.mse(N(x), t),
                                   rtol=1e-6)
        r = N.train(x, t, maxiter=200)
        np.testing.assert_allclose(r.lmbd, [1.])  # nothing to balance
        assert r.terms.shape == (1,) and r.nouter == 1
        assert r.fun < 1e-2

    def test_single_term_as_is(self):
        x, t = data()
        N = single([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, maxiter=50)
        np.testing.assert_allclose(r.fun, 2*N.mse(N(x), t), rtol=1e-5)

    def test_balance_grad(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, maxiter=300, balance=balance.grad(inner=100))
        assert r.lmbd.shape == (2,) and r.lmbd[0] == 1 and r.lmbd[1] >= 1
        assert r.nit <= 300 and r.nouter >= 1
        np.testing.assert_allclose(
            N._loss(N.w, jnp.asarray(r.lmbd), x, t), r.fun, rtol=1e-4)

    def test_balance_curv(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        # curvatures v.H.v of the terms against a full Hessian
        w = N.w
        v = jnp.ones_like(w).at[::2].set(-1.)
        H = jax.hessian(lambda w: N._terms(w, x, t))(w)
        np.testing.assert_allclose(balance._termsvhv(N, w, v[None], x, t)[0],
                                   jnp.einsum('i,kij,j->k', v, H, v),
                                   rtol=1e-3, atol=1e-6)
        b = balance.curv(inner=100)
        lm = b(N, w, None, x, t)
        assert lm.shape == (2,) and lm.min() == 1 and np.all(lm >= 1)
        r = N.train(x, t, maxiter=300, balance=b)   # 'curv' is the default
        assert r.nit <= 300 and np.all(r.lmbd >= 1)
        np.testing.assert_allclose(
            N._loss(N.w, jnp.asarray(r.lmbd), x, t), r.fun, rtol=1e-4)

    def test_curv_zero_curvature(self):
        class flat(spinn):
            @jax.jit
            def terms(self, w, x, t):
                self.w = jnp.asarray(w)
                return self.mse(self(x), t), jnp.asarray(0.5)  # w-free term
        x, t = data()
        N = flat([1, 8, 1], f=jnp.tanh, key=1)
        lm = balance.curv()(N, N.w, None, x, t)
        np.testing.assert_allclose(lm, [1., 1.])
        assert balance.curv.lmax == 1e12 and balance.sens.lmax == 1e12

    def test_cg(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((10, 10))
        A = M @ M.T + 10*np.eye(10)
        B = rng.standard_normal((3, 10))
        X = balance.cg(lambda V: V @ A, B, maxcg=100, tol=1e-12)
        np.testing.assert_allclose(X, np.linalg.solve(A, B.T).T, rtol=1e-8)
        # negative curvature: the (zero) solution so far is kept
        X = balance.cg(lambda V: -V, B, maxcg=10)
        np.testing.assert_allclose(X, 0.)

    def test_solution_domain(self):
        N = poisson([1, 8, 2], f=jnp.tanh, key=1)
        x1 = jnp.linspace(0., .5, 5)[:, None]
        x2 = jnp.linspace(.5, 1., 7)[:, None]
        w = N.w
        u = balance._sol(N, w, (x1, x2))
        np.testing.assert_allclose(
            u, jnp.concatenate([N(x1)[:, 0], N(x2)[:, 1]]), rtol=1e-6)
        sol, fields = balance._sol, balance._solfields
        np.testing.assert_allclose(sol(N, w, x1), N(x1).ravel(), rtol=1e-6)
        np.testing.assert_allclose(sol(N, w, (None, x2)), N(x2)[:, 1],
                                   rtol=1e-6)
        assert list(fields(N, (x1, x2))) == [0]*5 + [1]*7
        assert list(fields(N, x1[:2])) == [0, 1, 0, 1]
        vs = jnp.eye(w.size)[:3]
        J = jax.jacfwd(lambda w: sol(N, w, (x1, x2)))(w)
        np.testing.assert_allclose(balance._soljvp(N, w, (x1, x2), vs),
                                   vs @ J.T, rtol=1e-4, atol=1e-6)
        try:
            balance._sol(N, w, (x1,))
        except ValueError:
            return
        assert False

    def test_balance_sens(self):
        x = (jnp.linspace(0., .5, 21)[:, None],
             jnp.linspace(.5, 1., 21)[:, None])
        bx = jnp.asarray([[0.], [1.]])
        N = poisson([1, 16, 2], f=jnp.tanh, key=1)
        # Hessian-vector products against a full Hessian
        lm = jnp.asarray([1., 2., 3.])
        H = jax.hessian(lambda w: N._loss(w, lm, x, bx))(N.w)
        vs = jnp.eye(N.w.size)[:2]
        np.testing.assert_allclose(balance._losshv(N, N.w, lm, vs, x, bx),
                                   vs @ H, rtol=1e-3, atol=1e-4)
        b = balance.get('curv, curv**.5*sens')
        b.inner = 200
        w0 = N.w
        r = N.train(x, bx, maxiter=3000, balance=b)     # curv, then sens
        assert r.lmbd.shape == (3,) and r.lmbd.min() == 1
        assert np.all(np.isfinite(r.lmbd)) and np.all(r.lmbd <= 1e12)
        xs = jnp.concatenate(x)
        u = balance._sol(N, N.w, x)
        assert np.max(np.abs(u - jnp.sin(jnp.pi*xs).ravel())) < 1e-2

        # 'train' works on a copy: the instance is unchanged, and the same
        # training gives the same result
        assert b._k == 0 and b.phases[1].parts[1][0]._lf is None
        N.w = w0
        r2 = N.train(x, bx, maxiter=3000, balance=b)
        np.testing.assert_allclose(r2.lmbd, r.lmbd)

    def test_sens_zero_term(self):
        # a term at exactly 0 (float32) has a zero gradient: sensitivity 0/0;
        # it must not become the reference (all other weights at the cap)
        class zero(poisson):
            @jax.jit
            def terms(self, w, x, bx):
                T = poisson.terms(self, w, x, bx)
                return T[0], T[1], 0.*T[2]
        x = (jnp.linspace(0., .5, 11)[:, None],
             jnp.linspace(.5, 1., 11)[:, None])
        bx = jnp.asarray([[0.], [1.]])
        N = zero([1, 8, 2], f=jnp.tanh, key=2)
        P = poisson([1, 8, 2], f=jnp.tanh, key=2)
        b = balance.sens(avg='none')
        lm = b(N, N.w, None, x, bx)
        assert np.all(np.isfinite(lm)) and lm.max() < 1e6
        assert lm[:2].min() == 1 and lm[:2].min() <= lm[2] <= lm[:2].max()
        # a later update: the zero term keeps its previous log-sensitivity
        b = balance.sens(avg='none')
        b(P, N.w, None, x, bx)
        prev = b._lf.copy()
        lm = b(N, N.w, None, x, bx)
        assert np.all(np.isfinite(lm)) and lm[:2].min() == 1
        np.testing.assert_allclose(np.log(lm[2]), prev[2], rtol=1e-12)

    def test_sens_parts(self):
        x = (jnp.linspace(0., .5, 11)[:, None],
             jnp.linspace(.5, 1., 11)[:, None])
        bx = jnp.asarray([[0.], [1.]])
        N = poisson([1, 8, 2], f=jnp.tanh, key=2)
        w = N.w
        # sens against a dense solve (damping 2*max|eig| keeps H + mu*I
        # positive definite, so CG does not stop early)
        b = balance.sens(damp=2., maxcg=500, scale='none', avg='none')
        lm = b(N, w, None, x, bx)                 # equal weights in the loss
        H = np.asarray(jax.hessian(
            lambda w: N._loss(w, jnp.ones(3), x, bx))(w), np.float64)
        e = np.linalg.eigvalsh(H)
        G = np.asarray(balance._termsjac(N, w, x, bx), np.float64)
        dW = -np.linalg.solve(H + 2*np.abs(e).max()*np.eye(len(H)), G.T).T
        du = np.asarray(balance._soljvp(N, w, x, jnp.asarray(dW, w.dtype)))
        sv = np.linalg.norm(du, axis=1)/np.sqrt(np.asarray(N._terms(w, x, bx)))
        np.testing.assert_allclose(lm, sv/sv.min(), rtol=2e-2)
        # with the weights in the loss (set by 'train')
        lm0 = np.asarray([1., 2., 3.])
        lm2 = b(N, w, lm0, x, bx)
        H = np.asarray(jax.hessian(
            lambda w: N._loss(w, jnp.asarray([1., 2., 3.]), x, bx))(w),
            np.float64)
        e = np.linalg.eigvalsh(H)
        dW = -np.linalg.solve(H + 2*np.abs(e).max()*np.eye(len(H)),
                              (lm0[:, None]*G).T).T
        du = np.asarray(balance._soljvp(N, w, x, jnp.asarray(dW, w.dtype)))
        sv = np.linalg.norm(du, axis=1)/np.sqrt(np.asarray(N._terms(w, x, bx)))
        np.testing.assert_allclose(lm2, sv/sv.min(), rtol=2e-2)

        class const(balance.balancer):
            def __init__(self, v):
                super().__init__()
                self.v = v

            def __call__(self, model, w, lmbd, *args):
                return np.asarray(self.v, np.float64)
        p = balance.product((const([1., 4., 9.]), .5), const([2., 1., 1.]))
        np.testing.assert_allclose(p(N, w, None, x, bx), [1., 1., 1.5])
        p = balance.product((const([1., 1e10]), 1.), lmax=1e3)
        np.testing.assert_allclose(p(N, w, None, x, bx), [1., 1e3])
        p = balance.product(b, const([1., 1., 1.]))
        np.testing.assert_allclose(p(N, w, lm0, x, bx), lm2, rtol=1e-5)

    def test_balancer_phases(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)

        calls = []

        class two(balance.balancer):
            inner = 20
            phase = 0

            def advance(self):
                self.phase += 1
                return self.phase < 2

            def __call__(self, model, w, lmbd, *args):
                calls.append(self.phase)
                return [1., 2.] if self.phase == 0 else [1., 5.]
        r = N.train(x, t, maxiter=2000, balance=two())
        # phase 0 until its weights settle, then phase 1
        assert calls[0] == 0 and calls[-1] == 1
        assert sorted(calls) == calls and 0 in calls
        np.testing.assert_allclose(r.lmbd, [1., 5.])
        # hard stops (here 'outer', the number of optimizer runs) do not
        # advance, whatever the optimizer does
        calls.clear()
        r = N.train(x, t, maxiter=2000, balance=two(outer=1))
        assert set(calls) == {0} and r.nouter == 1

    def test_seeds(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, maxiter=200, seeds=[1, 2, 3])
        assert r.seeds == [1, 2, 3] and r.seed in r.seeds
        assert len(r.scores) == 3 and r.fun == r.scores.min()
        assert r.scores[r.seeds.index(r.seed)] == r.fun
        # the network keeps the weights of the best training
        np.testing.assert_allclose(
            N._loss(N.w, jnp.asarray(r.lmbd, N.w.dtype), x, t), r.fun,
            rtol=1e-5)
        # the same as one training from the seed kept
        M = fit2([1, 8, 1], f=jnp.tanh, key=r.seed)
        r1 = M.train(x, t, maxiter=200)
        np.testing.assert_allclose(r1.fun, r.fun, rtol=1e-4)
        # random seeds are reported
        r = N.train(x, t, maxiter=50, seeds=2)
        assert len(r.seeds) == 2 and all(type(k) is int for k in r.seeds)
        assert r.seed in r.seeds
        for bad in (0, []):
            with pytest.raises(ValueError):
                N.train(x, t, seeds=bad)

    def test_seeds_callback_stop(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)

        def cb(xk):
            raise StopIteration
        r = N.train(x, t, maxiter=500, seeds=[4, 5, 6], callback=cb)
        assert r.status == 99 and r.seeds == [4] and len(r.scores) == 1

    def test_seeds_exception(self):
        # an exception in a later training: raised, the network keeps the
        # best finished training
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, maxiter=100, seeds=[4])
        w4 = N.w
        n = []

        def cb(xk):
            n.append(1)
            if len(n) > 1:      # second seed
                raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            N.train(x, t, maxiter=100, seeds=[4, 5],
                    method=optimize.lbfgs(chunk=100), callback=cb)
        np.testing.assert_array_equal(N.w, w4)
        assert r.seed == 4

    def test_equal_and_fixed_weights(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, maxiter=5, balance=None)
        np.testing.assert_allclose(r.lmbd, [1., 1.])
        np.testing.assert_allclose(r.fun, jnp.sum(N._terms(N.w, x, t)),
                                   rtol=1e-5)
        r = N.train(x, t, maxiter=5, balance=[1., 3.])
        np.testing.assert_allclose(r.lmbd, [1., 3.])
        T = N._terms(N.w, x, t)
        np.testing.assert_allclose(r.fun, T[0] + 3*T[1], rtol=1e-5)
        try:
            N.train(x, t, maxiter=5, balance=[1., 3., 1.])
        except ValueError:
            return
        assert False

    def test_custom_rule(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)

        class rule(balance.balancer):
            a = 1.

            def __call__(self, model, w, lmbd, *args):
                return [1., self.a]
        r = N.train(x, t, maxiter=10, balance=rule(a=7., inner=5))
        np.testing.assert_allclose(r.lmbd, [1., 7.])
        assert r.nit <= 10 and 1 <= r.nouter <= 2

    def test_balancer_resolution(self):
        get = balance.get
        assert type(get('grad')) is balance.grad
        assert type(get('curv')) is balance.curv
        assert type(get('sens')) is balance.sens
        # expressions: '*' product, '**' power, ',' phases
        b = get('curv, curv**.5*sens')
        assert type(b) is balance.phases and len(b.phases) == 2
        assert type(b.phases[0]) is balance.curv
        (c, a1), (s1, a2) = b.phases[1].parts
        assert type(c) is balance.curv and type(s1) is balance.sens
        assert a1 == .5 and a2 == 1.
        b = get(' (curv*grad)**2 * sens**-1 ')
        assert [(type(p).__name__, a) for p, a in b.parts] == \
            [('curv', 2.), ('grad', 2.), ('sens', -1.)]
        b = get('curvsens')          # shortcut of 'curv, curv**.5*sens'
        assert type(b) is balance.phases
        assert type(b.phases[0]) is balance.curv
        assert [(type(p), a) for p, a in b.phases[1].parts] == \
            [(balance.curv, .5), (balance.sens, 1.)]
        b = get('curv, sens, grad')
        assert [type(p) for p in b.phases] == [balance.curv, balance.sens,
                                               balance.grad]
        assert get('curv**1') is not None
        for bad in ['curv**sens', 'curv+sens', 'curv,', 'fixed', 'product',
                    'curvsens*grad',
                    'curv()', '(curv, sens)*grad', 'import os', '']:
            try:
                get(bad)
            except ValueError:
                continue
            assert False, bad
        assert type(get(balance.grad)) is balance.grad
        b = balance.grad(lmax=10.)
        assert get(b) is b and b.lmax == 10. and balance.grad.lmax == 1e6
        assert type(get(None)) is balance.fixed and get(None).lmbd is None
        np.testing.assert_allclose(get([1, 2]).lmbd, [1, 2])
        for bad, err in [('nope', ValueError), ('balancer', ValueError),
                         (lambda m, w: [1.], TypeError)]:
            try:
                get(bad)
            except err:
                continue
            assert False, bad
        try:
            balance.grad(nope=1)
        except TypeError:
            return
        assert False

    def test_old_interface(self):
        import warnings

        class old(spinn):         # 'loss' is the old name of 'terms'
            @jax.jit
            def loss(self, w, x, t):
                self.w = jnp.asarray(w)
                return 2*self.mse(self(x), t)

        class both(single):
            def loss(self, w, x, t):
                return 0.
        x, t = data()
        with warnings.catch_warnings(record=True) as wr:
            warnings.simplefilter('always')
            N = old([1, 8, 1], f=jnp.tanh, key=1)
            r = N.train(x, t, maxiter=50)
            M = single([1, 8, 1], f=jnp.tanh, key=1)
            r2 = M.train(x, t, maxiter=50)
            np.testing.assert_allclose(r.fun, r2.fun, rtol=1e-6)
            np.testing.assert_allclose(N.w, M.w, rtol=1e-6)
            assert issubclass(wr[0].category, FutureWarning)
            assert "rename it to 'terms'" in str(wr[0].message)
            assert len(wr) == 1
            B = both([1, 8, 1], f=jnp.tanh, key=1)
            r3 = B.train(x, t, maxiter=50)
            np.testing.assert_allclose(r3.fun, r2.fun, rtol=1e-6)
            assert "'loss' is ignored" in str(wr[1].message)

        class con(spinn):
            def eqc(self, w, x, t):
                return 0.
        N = con([1, 4, 1], key=1)
        try:
            N.train(x, t, maxiter=1)
        except AttributeError:
            return
        assert False

    def test_pickle_after_train(self, tmp_path):
        import pickle
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        N.train(x, t, maxiter=5, balance=[1., 2.])
        M = pickle.loads(pickle.dumps(N))
        np.testing.assert_allclose(M(x), N(x))
        fname = str(tmp_path/'n.pkl')
        N.dump(fname)
        M = fit2.load(fname)
        np.testing.assert_allclose(M(x), N(x))


class gd(optimize.optimizer):
    """Plain gradient descent, own optimizer for the tests."""
    step = 0.05

    def __call__(self, model, w, lmbd, *args, n, callback=None):
        for i in range(n):
            w = w - self.step*model._lossg(w, lmbd, *args)
        return optimize.OptimizeResult(
            x=w, fun=float(model._loss(w, lmbd, *args)), nit=n, nfev=n,
            status=1, success=False)


class TestOptimize:
    def test_resolution(self):
        o = optimize.get('BFGS')
        assert isinstance(o, optimize.minimize) and o.method == 'BFGS'
        assert optimize.get(optimize.minimize).method == 'L-BFGS-B'
        o = optimize.minimize(tol=1e-6)
        assert optimize.get(o) is o
        with pytest.raises(TypeError):
            optimize.minimize(lr=1.)
        with pytest.raises(TypeError):
            optimize.get(len)
        assert isinstance(optimize.get('lbfgs'), optimize.lbfgs)
        assert isinstance(optimize.get('LBFGS'), optimize.lbfgs)

    def test_name_and_instance_agree(self):
        x, t = data()
        r = []
        for m in ('BFGS', optimize.minimize(method='BFGS', tol=1e-8)):
            N = spinn([1, 8, 1], f=jnp.tanh, key=1)
            r.append(N.train(x, t, method=m, tol=1e-8, maxiter=100))
        assert r[0].fun == r[1].fun and r[0].nit == r[1].nit

    def test_options_not_changed(self):
        # train passes its own arguments to a copy of the optimizer
        x, t = data()
        o = optimize.minimize(options={'maxcor': 5})
        N = spinn([1, 8, 1], f=jnp.tanh, key=1)
        N.train(x, t, method=o, tol=1e-6, options={'ftol': 0.},
                maxiter=50)
        assert o.tol == 1e-10 and o.options == {'maxcor': 5}

    def test_callback_stop(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)

        calls = []

        def cb(intermediate_result):
            calls.append(1)
            if len(calls) >= 5:
                raise StopIteration
        r = N.train(x, t, method='L-BFGS-B', callback=cb, maxiter=500)
        assert r.status == 99 and r.nouter == 1 and r.nit == 5

    def test_own_optimizer(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        f0 = N._loss(N.w, jnp.ones(2), x, t)
        r = N.train(x, t, method=gd, balance=None, maxiter=200)
        assert r.nit == 200 and r.fun < f0
        np.testing.assert_allclose(N._loss(N.w, jnp.ones(2), x, t), r.fun,
                                   rtol=1e-5)
        with pytest.raises(TypeError):  # scipy arguments
            N.train(x, t, method=gd, options={'maxcor': 5})

    def test_lbfgs(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        T0 = np.asarray(N._terms(N.w, x, t))
        r = N.train(x, t, method='lbfgs', maxiter=2000)
        assert np.all(r.terms < 0.1*T0) and r.nit <= 2000
        # the best weights of the last run are kept
        np.testing.assert_allclose(
            N._loss(N.w, jnp.asarray(r.lmbd, N.w.dtype), x, t), r.fun,
            rtol=1e-5)

    def test_lbfgs_limits(self):
        x, t = data()
        N = spinn([1, 8, 1], f=jnp.tanh, key=1)
        f0 = float(N._loss(N.w, jnp.ones(1), x, t))
        r = N.train(x, t, method='lbfgs', maxiter=7)
        assert r.nit == 7 and r.status == 1 and not r.success
        assert r.fun <= f0
        r = N.train(x, t, method=optimize.lbfgs(tol=1e-2), maxiter=500)
        assert r.status == 0 and r.success and r.nit < 500

    def test_lbfgs_callback(self):
        x, t = data()
        N = spinn([1, 8, 1], f=jnp.tanh, key=1)
        seen = []

        def cb(intermediate_result):
            seen.append(intermediate_result.nit)
            if len(seen) == 2:
                raise StopIteration
        r = N.train(x, t, method=optimize.lbfgs(chunk=3, tol=0.),
                    callback=cb, maxiter=100)
        assert seen == [3, 6] and r.status == 99 and r.nit == 6
        xs = []
        N.train(x, t, method=optimize.lbfgs(chunk=4, tol=0.),
                callback=xs.append, maxiter=10)
        assert len(xs) == 3 and xs[0].shape == N.w.shape

    def test_lbfgs_compiles_once(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        N.train(x, t, method='lbfgs', maxiter=50)
        n = optimize._lbfgs_run._cache_size()
        for key in (2, 3):  # new networks, balanced: several runs each
            N = fit2([1, 8, 1], f=jnp.tanh, key=key)
            r = N.train(x, t, method='lbfgs', maxiter=600,
                        balance=balance.curv(inner=100))
        assert r.nouter > 1 and optimize._lbfgs_run._cache_size() == n

    def test_adam(self):
        x, t = data()
        assert isinstance(optimize.get('Adam'), optimize.adam)
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        T0 = np.asarray(N._terms(N.w, x, t))
        r = N.train(x, t, method=optimize.adam(lr=1e-2), maxiter=2000,
                    balance=balance.curv(inner=500))
        assert np.all(r.terms < 0.1*T0) and r.nit == 2000
        # the best weights of the last run are kept
        np.testing.assert_allclose(
            N._loss(N.w, jnp.asarray(r.lmbd, N.w.dtype), x, t), r.fun,
            rtol=1e-5)
        r = N.train(x, t, method=optimize.adam(patience=3, lr=10.),
                    maxiter=500)
        assert r.status == 0 and r.nit < 500

    def test_optimizer_defaults(self):
        # 'inner' and 'maxiter' come from the optimizer unless set
        assert optimize.lbfgs.inner == 500 and optimize.lbfgs.maxiter == 15000
        assert optimize.adam.inner == 2000 and optimize.adam.maxiter == 30000
        assert balance.curv.inner is None
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, method=optimize.adam(inner=20, maxiter=50))
        assert r.nit == 50 and r.nouter == 3
        r = N.train(x, t, method=optimize.adam(inner=20, maxiter=50),
                    balance=balance.curv(inner=10))
        assert r.nit == 50 and r.nouter == 5
        r = N.train(x, t, method=optimize.adam(inner=20), maxiter=30)
        assert r.nit == 30 and r.nouter == 2

    def test_anytime(self):
        # adam's runs do not converge: settled weights move to the next
        # phase, and the last phase runs until maxiter
        x, t = data()
        calls = []

        class const(balance.balancer):
            def __init__(self, v, **options):
                super().__init__(**options)
                self.v = v

            def __call__(self, model, w, lmbd, *args):
                calls.append(self.v[1])
                return self.v
        b = balance.phases(const([1., 2.]), const([1., 5.]), inner=20)
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, method='adam', maxiter=200, balance=b)
        assert calls[:2] == [2., 2.] and set(calls[2:]) == {5.}
        assert r.nit == 200 and r.nouter == 10
        np.testing.assert_allclose(r.lmbd, [1., 5.])
        # not anytime: no run converges, the first phase is kept
        calls.clear()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, method=optimize.adam(anytime=False), maxiter=200,
                    balance=b)
        assert set(calls) == {2.} and r.nit == 200

        # 'advance' returns False only once, so a balancer which changes its
        # state in 'advance' anyway does not move past its last phase
        class two(balance.balancer):
            inner = 20
            phase = 0

            def advance(self):
                self.phase += 1
                return self.phase < 2

            def __call__(self, model, w, lmbd, *args):
                calls.append(self.phase)
                return [1., 2.] if self.phase == 0 else [1., 5.]
        calls.clear()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, method='adam', maxiter=200, balance=two())
        assert calls[0] == 0 and max(calls) <= 2 and r.nit == 200

    def test_adam_schedule(self):
        # the schedule counts the iterations of the whole training, the
        # instance passed to train is not changed
        x, t = data()
        steps = []

        def lr(k):
            jax.debug.callback(steps.append, k)
            return 1e-3
        o = optimize.adam(lr=lr)
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, method=o, maxiter=30,
                    balance=balance.curv(inner=10))
        assert r.nouter == 3 and o._k == 0
        assert [int(k) for k in steps] == list(range(30))

    def test_adam_compiles_once(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        N.train(x, t, method='adam', maxiter=50)
        n = optimize._adam_run._cache_size()
        for key, lr in ((2, 1e-2), (3, 1e-3)):
            N = fit2([1, 8, 1], f=jnp.tanh, key=key)
            r = N.train(x, t, method=optimize.adam(lr=lr), maxiter=300,
                        balance=balance.curv(inner=100))
        assert r.nouter > 1 and optimize._adam_run._cache_size() == n


def swish(x):       # module-level activation, saved by its dotted path
    return x*jax.nn.sigmoid(x)


class TestActivations:
    def test_names(self):
        x, _ = data()
        assert spinn([1, 8, 1]).f == ['identity', 'softplus', 'identity']
        assert spinn([1, 8, 1], f=jnp.tanh).f[1] == 'tanh'
        assert spinn([1, 8, 1], f=jax.nn.gelu).f[1] == 'gelu'
        assert spinn([1, 8, 1], f=swish).f[1] == swish.__module__ + '.swish'
        N = spinn([1, 8, 1], f=['identity', 'sin', 'identity'], initx=x, key=1)
        assert N.f == ['linear', 'sin', 'identity']
        # names give the same network as the functions
        M = spinn([1, 8, 1], f=jnp.sin, initx=x, key=1)
        np.testing.assert_allclose(M(x), N(x))
        import pytest
        with pytest.raises(ValueError, match='unknown activation'):
            spinn([1, 8, 1], f='nosuchfunction')

    def test_dump_load(self, tmp_path):
        x, _ = data()
        fname = str(tmp_path/'n.pkl')
        for f, p in (('gelu', None), ('elu', (0.5,)), (swish, None)):
            N = spinn([1, 8, 1], f=f, p=p, initx=x, key=1)
            N.dump(fname)
            M = spinn.load(fname)
            assert M.f == N.f
            np.testing.assert_allclose(M(x), N(x))

    def test_lambda(self, tmp_path):
        import pytest
        x, _ = data()
        N = spinn([1, 8, 1], f=lambda x: jnp.sin(3*x), initx=x, key=1)
        np.testing.assert_allclose(N(x)[:, 0], N(x)[:, 0])   # works
        with pytest.raises(ValueError, match='layer 1 is a lambda'):
            N.dump(str(tmp_path/'n.pkl'))


if __name__ == "__main__":
    pytest.main()
