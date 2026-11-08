import jax
import jax.numpy as jnp
import numpy as np
from spinn import utils, graphs, spinn, balance

# jax.config.update('jax_default_matmul_precision', 'high')  # 'bfloat16_3x'


class TestSpinn:
    def test_simple_layered(self):
        ni, no = 5, 3
        arch = [ni, 30, 40, no]
        x = jnp.asarray(np.random.rand(10, ni))
        t = jnp.asarray(np.random.rand(10, no))
        G = graphs.mlgraph(arch)
        N = spinn(G, initx=x, inity=t)

        # Apply same weights
        w = utils.randomweights(arch)
        ww = [(wi[0].T.ravel(), wi[1]) for wi in w]
        wl, bl = list(zip(*ww))
        ww = jnp.concatenate(wl + bl)
        N.w = ww

        c = utils.to01c(x) + utils.from01c(t)
        y1 = np.array(N(x))
        y2 = np.array(utils.simpleprop(w, x, c))
        print(y1-y2)
        print(y2[0])

        np.testing.assert_almost_equal(y1, y2, 5)


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


def data():
    x = jnp.linspace(0., 1., 20)[:, None]
    return x, jnp.sin(3*x)


class TestTrain:
    def test_default_terms(self):
        x, t = data()
        N = spinn([1, 8, 1], f=jnp.tanh, key=1)
        np.testing.assert_allclose(N.loss(N.w, x, t), N.mse(N(x), t),
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
        np.testing.assert_allclose(N.lmbd, r.lmbd)
        np.testing.assert_allclose(N.loss(N.w, x, t, lmbd=r.lmbd), r.fun,
                                   rtol=1e-4)

    def test_equal_and_fixed_weights(self):
        x, t = data()
        N = fit2([1, 8, 1], f=jnp.tanh, key=1)
        r = N.train(x, t, maxiter=5, balance=None)
        np.testing.assert_allclose(r.lmbd, [1., 1.])
        np.testing.assert_allclose(r.fun, N.loss(N.w, x, t), rtol=1e-5)
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

            def __call__(self, model, w, *args):
                return [1., self.a]
        r = N.train(x, t, maxiter=10, balance=rule(a=7., inner=5))
        np.testing.assert_allclose(r.lmbd, [1., 7.])
        assert r.nit <= 10 and 1 <= r.nouter <= 2

    def test_balancer_resolution(self):
        get = balance.get
        assert type(get('grad')) is balance.grad
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

    def test_old_interface_rejected(self):
        class own(spinn):
            def loss(self, w, x, t):
                return 0.

        class con(spinn):
            def eqc(self, w, x, t):
                return 0.
        x, t = data()
        for cls in (own, con):
            N = cls([1, 4, 1], key=1)
            try:
                N.train(x, t, maxiter=1)
            except AttributeError:
                continue
            assert False

    def test_pickle_after_train(self, tmp_path):
        import pickle
        x, t = data()
        N = fit2([1, 8, 1], key=1)    # (jnp.tanh cannot be pickled)
        N.train(x, t, maxiter=5, balance=[1., 2.])
        M = pickle.loads(pickle.dumps(N))
        np.testing.assert_allclose(M.lmbd, [1., 2.])
        fname = str(tmp_path/'n.pkl')
        N.dump(fname)
        M = fit2.load(fname)
        np.testing.assert_allclose(M(x), N(x))


if __name__ == "__main__":
    import pytest
    pytest.main()
