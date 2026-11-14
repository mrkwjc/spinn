"""
Monitoring trainings: the beam of 'beam.py', with the error against its
exact solution recorded as a metric.

A monitor records terms, loss, lambdas and own metrics during training
(every 100 iterations for L-BFGS, 500 for Adam, and at each lambda update),
together with the settings and results of each training. One monitor can
record several trainings, e.g. to compare optimizers or balancers, and with
'seeds' it records every seed. A training stopped early (here by its
iteration limit) is continued by 'resume', and its monitor records on.
"""
import jax.numpy as jnp
from spinn import hcube, mlgraph, monitor, balance

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

    # 0. Warm-up: the first training of each kind compiles its code (a few
    # seconds), which the monitor would count as training time. A few
    # iterations of each optimizer, the sensitivities of 'curvsens' and the
    # metric compile it beforehand (for this network and these data), so
    # that the times below are of the trainings alone
    B = beam(mlgraph((1, 32, 1)), initx=x, key=1)
    for method in ('lbfgs', 'adam'):
        B.train(x, q, bx, EI, method=method, maxiter=10)
    balance.sens()(B, B.w, None, x, q, bx, EI)
    err(B)

    # 1. One monitor for four trainings: two optimizers, two balancers
    m = monitor(metrics={'err': err})
    for method in ('lbfgs', 'adam'):
        for bal in ('curv', 'curvsens'):
            B = beam(mlgraph((1, 32, 1)), initx=x, key=1)
            B.train(x, q, bx, EI, method=method, balance=bal, monitor=m)

    print('%-6s %-9s %8s %7s %6s %10s' % ('method', 'balance', 'time [s]',
                                          'nit', 'runs', 'error'))
    h = m.history
    for i in m.info:
        e = h['err'][h['train'] == i['train']][-1]
        print('%-6s %-9s %8.2f %7d %6d %10.1e' % (
            i['method'], i['balance'], i['time'], i['nit'], i['nouter'], e))

    # 2. Best of three initial weights; every seed is recorded
    B = beam(mlgraph((1, 32, 1)), initx=x)
    r = B.train(x, q, bx, EI, seeds=3, monitor=monitor(metrics={'err': err}))
    hs = r.monitor.history
    for k in r.seeds:
        print('seed %10d  error %.1e%s' % (
            k, hs['err'][hs['seed'] == k][-1],
            '  <- kept (lowest loss)' if k == r.seed else ''))

    # 3. Adam with 10000 iterations, then 20000 more by 'resume': it ends
    # where the training of part 1 with 30000 iterations at once ended
    B = beam(mlgraph((1, 32, 1)), initx=x, key=1)
    ra = B.train(x, q, bx, EI, method='adam', balance='curvsens',
                 maxiter=10000, monitor=monitor(metrics={'err': err}))
    print('%-34s error %.1e' % ('adam, 10000 iterations:', err(B)))
    ra = B.resume(ra, maxiter=20000)
    print('%-34s error %.1e' % ('resumed, 30000 iterations:', err(B)))
    print('%-34s error %.1e' % ('part 1, 30000 iterations at once:',
                                h['err'][h['train'] == 3][-1]))

    #
    # Plots
    import matplotlib.pyplot as plt

    # the course of one training: terms, loss, lambdas (dashed lines: lambda
    # updates) and the error
    m.plot(train=1, labels=['equation', 'deflection BC', 'slope BC'])
    # the error of all four trainings against time and iterations
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    m.compare('err', ax=ax[0])
    m.compare('err', x='it', ax=ax[1])
    # the seeds
    r.monitor.compare('err', x='it')
    # the resumed training: one history, the red dash-dot line is the resume
    ra.monitor.plot(labels=['equation', 'deflection BC', 'slope BC'])

    plt.show()
