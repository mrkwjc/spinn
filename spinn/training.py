"""
Training of spinn networks: the loop of 'spinn.train' and 'spinn.resume'.

A trainer is one training: its data, optimizer and balancer (with their
state), the lambdas, the phase of the balancer and the counters. 'train'
makes a fresh one, 'fit' runs it; the result keeps it as 'res.state', and
'resume' fits a copy of it further:

    t = trainer((x, bx), method='adam', balance='curvsens')
    res = t.fit(P, maxiter=10000)       # trains P, res.state is t
    res = t.fit(P, maxiter=5000)        # 5000 iterations more (t changes)

Usually 'P.train' and 'P.resume' are enough; see their help for the options.
"""
import copy

import jax.numpy as jnp
import numpy as np

from spinn import balance as balancing
from spinn import optimize as optimizers
from spinn import monitor as monitoring
from spinn.result import result as _result

__all__ = ['trainer', 'train', 'resume']


def _monitor(m):
    """Monitor from the 'monitor' argument: True the default one, a number
    'every', False or None no monitor."""
    if m is True:
        return monitoring.monitor()
    if m is False or m is None:
        return None
    if isinstance(m, (int, np.integer)):
        if m < 1:
            raise ValueError("monitor: 'every' must be positive")
        return monitoring.monitor(every=int(m))
    return m


class trainer:
    """
    One training of a network on the data 'args' (the arguments of its
    'terms'), with the optimizer 'method' and the balancer 'balance' (as in
    'spinn.train'; the trainer gets its own copies, which keep their
    state). 'tol' sets the tolerance of the optimizer, other keyword
    arguments go to a scipy optimizer. 'maxiter' (default the optimizer's)
    and 'seed' are only recorded, for the monitor and 'res.seed'.

    'fit' trains a network and continues on each call. Attributes: 'opt'
    and 'b' (optimizer, balancer), 'lmbd' (None before the first fit), 'x'
    (the final weights), 'nit', 'nfev', 'k' (runs of the optimizer),
    'phase' (of the balancer), 'resume' (how many times it was resumed),
    'monitor' (the last one that recorded it).
    """
    def __init__(self, args=(), method='lbfgs', balance='curv', tol=None,
                 maxiter=None, seed=None, **kwargs):
        opt = copy.deepcopy(optimizers.get(method))  # state of this training
        if tol is not None:
            opt.tol = tol
        if kwargs:
            if not isinstance(opt, optimizers.minimize):
                raise TypeError("%s got unexpected arguments %s"
                                % (type(opt).__name__, list(kwargs)))
            options = kwargs.pop('options', None) or {}
            opt.options = dict(opt.options or {}, **options)
            opt.kwargs = dict(opt.kwargs or {}, **kwargs)
        self.args, self.opt = tuple(args), opt
        self.b = copy.deepcopy(balancing.get(balance))
        self.method, self.balance, self.maxiter = method, balance, maxiter
        self.seed = seed
        self.lmbd = None        # fresh: set by the first lambdas
        self.x = None           # final weights
        self.nit = self.nfev = self.k = self.short = self.phase = 0
        self.last = False       # the balancer has no next phase
        self.pending = False    # lambda update after the last run to do
        self.lastrun = None     # (its iterations, success)
        self.resume = 0         # number of resumes so far
        self.monitor = None     # monitor which recorded it
        self.mtrain = None      # its training number in that monitor

    def __repr__(self):
        return ('<training state: %d runs, %d iterations, phase %d, '
                'resumed %d times>' % (self.k, self.nit, self.phase,
                                       self.resume))

    def resumed(self, args=(), tol=None):
        """A copy to go on with (for 'resume'; this trainer stays as it
        is): new 'args' replace the data, 'tol' that of the optimizer."""
        t = copy.copy(self)
        t.opt, t.b = copy.deepcopy(self.opt), copy.deepcopy(self.b)
        t.resume = self.resume + 1
        if args:
            t.args = tuple(args)
        elif t.args is None:
            raise ValueError("the data of the training were not saved with "
                             "the result: pass them, resume(res, *args)")
        if getattr(t.opt, 'lr', 0.) is None:
            raise ValueError("the learning rate schedule of adam was not "
                             "saved (no importable name): set it, "
                             "res.state.opt.lr = schedule")
        if tol is not None:
            t.opt.tol = tol
        return t

    def _terms(self, model, w):
        return np.asarray(model._terms(w, *self.args), np.float64)

    def _lambdas(self, model, w, current):  # current: lambdas in the loss
        return np.asarray(self.b(model, w, current, *self.args), np.float64)

    def _decide(self, model, w, rnit, success):
        """After a run of 'rnit' iterations: new lambdas, next phase; True
        if the training is done."""
        # no progress
        done = rnit == 0 or self.short >= 2
        settled = False
        new = self.lmbd
        if not done:
            new = self._lambdas(model, w, self.lmbd)
            settled = np.max(np.abs(new/self.lmbd - 1.)) < 0.1
            # converged and lambdas settled
            done = success and settled
        # an 'anytime' optimizer (e.g. adam, its runs do not converge) moves
        # to the next phase once the lambdas settle, and in the last phase
        # goes on
        if done or (settled and self.opt.anytime and not self.last):
            if not self.last and self.b.advance():  # next phase, if any
                self.short = 0
                self.phase += 1
                new = self._lambdas(model, w, self.lmbd)
            elif done:
                return True
            else:  # never asked again: 'advance' returns False once
                self.last = True
        self.lmbd = new
        return False

    def fit(self, model, maxiter=None, disp=False, callback=None,
            monitor=None):
        """
        Train 'model' for at most 'maxiter' further iterations (default the
        optimizer's): from its weights on the first call, from the final
        weights 'x' after that. The network gets the final weights; the
        result ('spinn.result') has this trainer as 'res.state'. 'disp' and
        'callback' as in 'spinn.train'; 'monitor' records (a monitor, a
        number 'every' or True); None goes on with the last one, False
        records nothing.
        """
        opt = self.opt
        fresh = self.lmbd is None
        w = np.asarray(model.w, np.float64) if fresh else self.x
        if maxiter is None:
            maxiter = opt.maxiter
        n0, end = maxiter, self.nit + maxiter    # further, total
        mon = _monitor(self.monitor if monitor is None else monitor)
        if mon is not None:
            every = opt.every if mon.every is None else mon.every
            if hasattr(opt, 'chunk'):  # optimizers on the device
                opt.chunk = every
            cb = callback
            state = {}  # of the run: iterations before it, lambdas, run...

            def callback(intermediate_result):  # noqa: F811
                r = intermediate_result
                state['calls'] += 1
                # optimizers on the device give the iteration of the run,
                # scipy calls back every iteration
                it = state['nit'] + getattr(r, 'nit', state['calls'])
                # (at the end of the run its own record follows)
                if it - state['last'] >= every and it < state['end']:
                    state['last'] = it
                    mon._record(r.x, it, state['run'], state['phase'],
                                state['lmbd'])
                if cb is not None:
                    optimizers._call(cb, r.x, getattr(r, 'fun', None), it)
                if mon._timeout():
                    raise StopIteration

        dtype = jnp.asarray(model.w).dtype
        if fresh:
            if len(self._terms(model, w)) < 2:
                self.b = balancing.fixed()  # one term, nothing to balance
            self.lmbd = self._lambdas(model, w, None)
        b = self.b
        inner = opt.inner if b.inner is None else b.inner
        k0 = self.k
        if mon is not None:
            on = mon is self.monitor and not fresh  # the same history goes on
            mon._seed = self.seed
            mon._begin(model, self.args, opt, b, method=self.method,
                       balance=self.balance, resume=self.resume,
                       train=self.mtrain if on else None,
                       maxiter=self.maxiter if fresh else n0,
                       inner=inner, every=every)
            state['last'] = self.nit
            if not on:
                mon._record(w, self.nit, self.k, self.phase, self.lmbd)

        try:
            if self.pending:  # the update after the last run of the fit
                self._decide(model, w, *self.lastrun)  # (done or not: on)
                self.pending = False
            while self.nit < end:
                n = int(min(inner, end - self.nit))
                if mon is not None:
                    state.update(calls=0, nit=self.nit, run=self.k + 1,
                                 lmbd=self.lmbd, phase=self.phase,
                                 end=self.nit + n)
                res = opt(model, w, jnp.asarray(self.lmbd, dtype),
                          *self.args, n=n, callback=callback)
                w = res.x
                self.nit += res.nit
                self.nfev += res.nfev
                self.k += 1
                if mon is not None:  # end of the run: the lambdas change
                    state['last'] = self.nit
                    mon._record(w, self.nit, self.k, self.phase, self.lmbd,
                                end=True)
                if disp:
                    fmt = {'float_kind': lambda v: '%.2e' % v}
                    print('run %3d: nit %5d  terms %s  lmbd %s'
                          % (self.k, res.nit,
                             np.array2string(self._terms(model, w),
                                             formatter=fmt),
                             np.array2string(self.lmbd, formatter=fmt)))
                # (almost) no progress twice in a row: line search failed,
                # or the lambdas keep changing while the optimizer has
                # nothing left to do (once is not enough, a run with poor
                # lambdas can stop early and the next lambdas recover)
                self.short = self.short + 1 if res.nit < min(10, n) else 0
                # stopped by callback, out of iterations or lambda updates:
                # the update after this run is left to the next fit
                if res.status == 99 or self.nit >= end or self.k >= b.outer:
                    self.pending = True
                    self.lastrun = (res.nit, res.success)
                    break
                if self._decide(model, w, res.nit, res.success):
                    break
        except BaseException:
            if mon is not None:
                mon._seed = None
            raise
        if self.k == k0:  # no run: nothing left to do
            res = optimizers.OptimizeResult(
                x=w, fun=float(np.sum(self.lmbd*self._terms(model, w))),
                status=1, success=False, message='no iterations left')
        model.w = jnp.asarray(w)
        self.x = w
        res = _result(res)
        res['_model'] = copy.copy(model)  # for res.model (at res.x)
        res.nit, res.nfev, res.nouter = self.nit, self.nfev, self.k
        res.lmbd = self.lmbd
        res.terms = self._terms(model, w)
        self.monitor = mon
        if mon is not None:
            self.mtrain = mon._cur
            mon._end(w, res)
            mon._seed = None
            res.monitor = mon
        res.state = self
        return res


def train(model, args, tol=None, maxiter=None, disp=False, method='lbfgs',
          balance='curv', seeds=None, monitor=None, callback=None,
          seed=None, **kwargs):
    """'spinn.train' (see its help): a fresh trainer fitted to 'model'."""
    monitor = _monitor(monitor)
    if seeds is not None:
        return _seeds(model, seeds, args, dict(
            tol=tol, maxiter=maxiter, disp=disp, method=method,
            balance=balance, monitor=monitor, callback=callback, **kwargs))
    t = trainer(args, method=method, balance=balance, tol=tol,
                maxiter=maxiter, seed=seed, **kwargs)
    return t.fit(model, maxiter, disp, callback, monitor)


def resume(model, res, args=(), maxiter=None, tol=None, disp=False,
           callback=None, monitor=None):
    """'spinn.resume' (see its help): a copy of 'res.state' fitted on."""
    t = res.get('state') if hasattr(res, 'get') else None
    if not isinstance(t, trainer):
        raise TypeError("res must be a result of 'train' or 'resume'")
    return t.resumed(args, tol).fit(model, maxiter, disp, callback, monitor)


def _seeds(model, seeds, args, kw):
    """'train' from each seed, keep the lowest 'res.fun'."""
    if isinstance(seeds, (int, np.integer)):
        if seeds < 1:
            raise ValueError("seeds must be a positive number or a list")
        seeds = np.random.default_rng().integers(2**31, size=seeds)
    seeds = [int(k) for k in seeds]
    if not seeds:
        raise ValueError("seeds must be a positive number or a list")
    best, scores = None, []
    try:
        for i, key in enumerate(seeds):
            if kw['disp']:
                print('seed %d (%d of %d)' % (key, i + 1, len(seeds)))
            model.initw(key)
            res = train(model, args, seed=key, **kw)
            score = float(res.fun)
            scores.append(score)
            if best is None or score < best[0] or not best[0] == best[0]:
                best = (score, key, model.w, res)
            if res.status == 99:  # stopped by callback
                break
    finally:  # also on an exception: the best training so far
        if best is not None:
            model.w = best[2]
    _, key, _, res = best
    res.seed, res.seeds, res.scores = key, seeds[:len(scores)], \
        np.array(scores)
    return res
