"""
Saving and loading results: the beam of 'beam.py'.

'res.save' stores a training in one file: the network, the result, the
monitor and the state for 'resume' (the data of the training only with
data=True). 'result.load' restores all; with model=False it needs nothing
but spinn: results, monitor and the network as a plain spinn ('network'),
e.g. on another computer, without the class of the problem. The files are
numpy NPZ archives, readable with numpy alone.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import jax.numpy as jnp
from spinn import hcube, mlgraph, monitor, result

from beam import beam, exact


if __name__ == "__main__":
    L, q0, EI = 4., 8., 1.
    x = hcube(L, n=401)
    q = -q0 * jnp.sin(jnp.pi*(x/L))
    bx = x[jnp.asarray([0, -1])]
    Fe = exact(x, q0, L, EI)[0]

    def err(B):
        """Relative L2 error of the deflection (metric of the monitor)."""
        F = B.propx([[x, 0]])[0]
        return jnp.linalg.norm(F - Fe) / jnp.linalg.norm(Fe)

    d = tempfile.mkdtemp()
    fname = os.path.join(d, 'beam.res')

    # 1. Train (Adam stopped early) and save
    B = beam(mlgraph((1, 32, 1)), initx=x, key=1)
    res = B.train(x, q, bx, EI, method='adam', balance='curvsens',
                  maxiter=10000, monitor=monitor(metrics={'err': err}))
    res.save(fname)                                        # without data
    res.save(os.path.join(d, 'beam_data.res'), data=True)  # with data
    print('trained:  %d iterations, error %.1e' % (res.nit, err(res.model)))

    # 2. Light load: results, monitor, network as a plain spinn
    r = result.load(fname, model=False)
    print('light:    loss %.3e, lambdas %s, network %s with %d weights, '
          'error %.1e' % (r.fun, np.array2string(r.lmbd, precision=2),
                          type(r.network).__name__, r.network.w.size,
                          err(r.network)))

    # 3. Full load: the network as a beam, continued by resume (the data
    # were not saved: pass them; the metric function was not saved either)
    r = result.load(fname)
    r.monitor.metrics['err'] = err
    P = r.model                     # keep the network to work with it
    r2 = P.resume(r, x, q, bx, EI, maxiter=20000)
    print('resumed:  %d iterations, error %.1e' % (r2.nit, err(P)))

    # the file with data resumes without them
    r = result.load(os.path.join(d, 'beam_data.res'))
    r.monitor.metrics['err'] = err
    r3 = r.model.resume(r, maxiter=20000)
    print('resumed:  %d iterations, error %.1e (data from the file)'
          % (r3.nit, err(r3.model)))

    # 4. The same file with numpy alone
    with np.load(fname) as z:
        meta = json.loads(str(z['__meta__']))
    print('numpy:    %s of spinn %s, network %s, %d arrays'
          % (meta['format'], meta['spinn'], meta['cls'], len(z.files) - 1))

    # 5. The network alone: save and load
    P.save(os.path.join(d, 'beam.net'))
    Q = beam.load(os.path.join(d, 'beam.net'))
    print('network:  error %.1e after save and load' % err(Q))
    shutil.rmtree(d)                # the files of this example

    #
    # Plots: the history of the training continued after loading
    import matplotlib.pyplot as plt

    r2.monitor.plot(labels=['equation', 'deflection BC', 'slope BC'])
    plt.show()
