import jax
import jax.numpy as jnp
import numpy as np
from spinn import utils, graphs, spinn

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


if __name__ == "__main__":
    import pytest
    pytest.main()
