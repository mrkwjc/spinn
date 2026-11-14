"""
Monitoring trainings of a 2D problem: the Laplace equation of
'laplace.py', with the error against its exact solution on a finer grid
(points not seen in training) recorded as a metric.

Two optimizers (L-BFGS, Adam) with three ways of balancing the five loss
terms: 'curv', 'curvsens' and the fixed lambdas of 'laplace.py' (the
Dirichlet conditions weighted up). See 'monitor_beam.py' for the monitor
itself.
"""
import jax.numpy as jnp
from spinn import hcube, mlgraph, monitor, balance, left, right, bottom, top

from laplace import laplace, exact


if __name__ == "__main__":
    n, L, B = 21, 1., 1.
    x = hcube((L, B), n=n)
    bx = left(x), right(x), bottom(x), top(x)
    q = B/2 * (1 - jnp.cos(jnp.pi * bx[1][:, 1] / B))
    xt = hcube((L, B), n=101)  # test grid
    Fe = exact(xt, L, B)

    def err(P):
        """Relative L2 error on the test grid (metric of the monitor)."""
        return jnp.linalg.norm(P(xt).T[0] - Fe) / jnp.linalg.norm(Fe)

    def network():
        return laplace(mlgraph((2, 16, 16, 1)), initx=x, key=1)

    # 0. Warm-up: compile the optimizers, the sensitivities of 'curvsens'
    # and the metric beforehand, so that the times below are of the
    # trainings alone (see 'monitor_beam.py')
    P = network()
    for method in ('lbfgs', 'adam'):
        P.train(x, q, bx, method=method, maxiter=10)
    balance.sens()(P, P.w, None, x, q, bx)
    err(P)

    # 1. One monitor for six trainings: two optimizers, three balancings
    balancings = {'curv': 'curv', 'curvsens': 'curvsens',
                  'fixed': [1., 10., 10., 1., 1.]}
    m = monitor(metrics={'err': err})
    names = []
    for method in ('lbfgs', 'adam'):
        for name, bal in balancings.items():
            P = network()
            P.train(x, q, bx, method=method, balance=bal, monitor=m)
            names.append(name)

    print('%-6s %-9s %8s %7s %6s %10s  %s' % (
        'method', 'balance', 'time [s]', 'nit', 'runs', 'error', 'stop'))
    h = m.history
    for name, i in zip(names, m.info):
        e = h['err'][h['train'] == i['train']][-1]
        print('%-6s %-9s %8.2f %7d %6d %10.1e  %s' % (
            i['method'], name, i['time'], i['nit'], i['nouter'], e,
            i['message']))

    #
    # Plots
    import matplotlib.pyplot as plt

    labels = ['equation', 'F left', 'F right', 'dFy bottom', 'dFy top']
    # the error of all six trainings against time and iterations
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    m.compare('err', ax=ax[0])
    m.compare('err', x='it', ax=ax[1])
    # the course of two trainings with 'curvsens': terms, loss, lambdas
    # (dashed lines: lambda updates) and the error
    m.plot(train=1, labels=labels)     # lbfgs
    m.plot(train=4, labels=labels)     # adam

    plt.show()
