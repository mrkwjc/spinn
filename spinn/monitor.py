"""
Training monitor, used by 'spinn.train' (argument 'monitor').

A monitor records the course of a training: every 'every' iterations and
at the end of each optimizer run (when the lambdas are updated) it stores a
row with the iteration, time, unweighted terms, weighted loss, lambdas and
own metrics. It also keeps the final network weights of each training.

    from spinn import monitor
    m = monitor(every=100, metrics={'err': lambda P: error(P)}, tmax=600)
    res = P.train(x, bx, monitor=m)       # res.monitor is m
    res = P.train(x, bx, monitor=200)     # monitor(every=200), in res.monitor
    m.history['terms']       # numpy arrays, one row per record
    m.final                  # final weights of each training, by seed
    m.plot()

'history' has one entry per record (numpy arrays):

    train - number of the training with this monitor (0, 1, ...)
    seed  - seed of the training ('seeds' of train; -1 without)
    it    - optimizer iterations since the start of the training
    t     - seconds since the start of the training
    run   - optimizer run (0 at the start; the lambdas change between runs)
    phase - phase of the balancer (e.g. 0 curv, 1 sensitivities)
    end   - True at the end of a run (a lambda update)
    loss  - weighted loss, sum(lmbd * terms)
    terms - unweighted terms (one column per term)
    lmbd  - lambdas (one column per term)
    w     - network weights (with keepw=True)
    and one entry per metric.

'info' has one dict per training: 'train', 'seed', 'start' (date and
time), 'method' and 'balance' (as given) with 'optimizer' and 'balancer'
(all options), 'maxiter', 'inner' and 'every' (as used); at its end also
'time' (seconds), 'nit', 'nouter', 'fun', 'status' and 'message' (missing
if the training was interrupted).

A monitor pickles as plain data: history, final weights, info and
options. Metric functions are not kept (they may not pickle, or change
with the code): after loading their entries in 'metrics' are None, their
recorded values remain; set the functions again to train further with the
monitor. Metrics given as method names are kept. Trainings of problems
with fewer terms than others in one monitor have nan in the extra columns
of 'terms' and 'lmbd' (what term i means is up to the problem).

Unlike balancers and optimizers, 'train' does not copy the monitor: it is
the recorder passed in, readable also after an interruption (Ctrl+C, an
exception, 'tmax'). Several trainings with one monitor append to its
history (columns 'train' and 'seed' tell them apart); with 'seeds' each
seed is a training. Use a new monitor for a fresh history.

Options:

    every   - record every 'every' optimizer iterations (default None: the
              optimizer's, 100 for lbfgs and minimize, 500 for adam; for
              'lbfgs' and 'adam' this sets their 'chunk')
    metrics - {name: metric} evaluated at each record (default None): a
              function f(model), e.g. the error against an exact solution,
              or the name of a method of the model, e.g. {'err': 'error'}
              calls model.error(); a method of the model class given as a
              function (beam.error) is stored as its name
    tmax    - time limit of a training in seconds: the training stops as if
              by 'callback' (default None, no limit)
    disp    - print each record (default False)
    keepw   - also store the network weights at each record (default False;
              one weight vector per row)
"""
import time

import numpy as np

__all__ = ['monitor']
_COLUMNS = ('train', 'seed', 'it', 't', 'run', 'phase', 'end', 'loss',
            'terms', 'lmbd', 'w')


class monitor:
    __doc__ = __doc__
    every = None
    metrics = None
    tmax = None
    disp = False
    keepw = False

    def __init__(self, **options):
        for k, v in options.items():
            if not hasattr(self, k) or k.startswith('_'):
                raise TypeError("monitor has no option '%s'" % k)
            setattr(self, k, v)
        self.rows = []      # records, dicts
        self.final = {}     # final weights of each training, by seed
        self.info = []      # one dict per training: settings and result
        self._ntrain = -1
        self._seed = None   # set by train with 'seeds'

    def __repr__(self):
        return ('monitor(every=%r, metrics=%r, tmax=%r, disp=%r, keepw=%r; '
                '%d rows)' % (self.every, self.metrics and
                              list(self.metrics), self.tmax, self.disp,
                              self.keepw, len(self.rows)))

    # --- called by train -------------------------------------------------

    def _begin(self, model, args, opt, balancer, method, balance, **used):
        """Start of a training: 'used' are the values train runs with
        (maxiter, inner, every)."""
        if self.metrics:
            metrics = {}
            for name, f in self.metrics.items():
                if f is None:
                    raise ValueError(
                        "metric '%s' has no function (not kept when the "
                        "monitor was pickled); set m.metrics['%s'] = f"
                        % (name, name))
                if callable(f) and getattr(
                        type(model), getattr(f, '__name__', ''), None) is f:
                    f = f.__name__  # a method of the model class
                if isinstance(f, str) and not callable(
                        getattr(model, f, None)):
                    raise ValueError("metric '%s': %s has no method '%s'"
                                     % (name, type(model).__name__, f))
                if not isinstance(f, str) and not callable(f):
                    raise TypeError("metric '%s' must be a function or a "
                                    "method name" % name)
                metrics[name] = f
            self.metrics = metrics
        self._ntrain += 1
        self._t0 = time.time()
        self._model, self._args = model, args

        def name(m):
            return m if isinstance(m, str) or m is None else \
                type(m).__name__ if not isinstance(m, type) else m.__name__
        self.info.append(dict(
            train=self._ntrain, seed=self._seed,
            start=time.strftime('%Y-%m-%d %H:%M:%S'),
            method=name(method), optimizer=repr(opt),
            balance=name(balance) if not isinstance(
                balance, (list, tuple, np.ndarray)) else list(balance),
            balancer=repr(balancer), **used))

    def _record(self, w, it, run, phase, lmbd, end=False):
        """One row at network weights 'w' (numpy) after 'it' iterations of
        the training, in optimizer run 'run' with lambdas 'lmbd'."""
        model, args = self._model, self._args
        wj = np.asarray(w, np.asarray(model.w).dtype)
        T = np.asarray(model._terms(wj, *args), np.float64)
        lmbd = np.asarray(lmbd, np.float64)
        row = dict(train=self._ntrain,
                   seed=-1 if self._seed is None else self._seed,
                   it=it, t=time.time() - self._t0, run=run, phase=phase,
                   end=end, loss=float(np.sum(lmbd*T)), terms=T, lmbd=lmbd)
        if self.metrics:
            w0 = model.w
            model.w = wj
            try:
                for name, f in self.metrics.items():
                    row[name] = float(getattr(model, f)() if isinstance(
                        f, str) else f(model))
            finally:
                model.w = w0
        if self.keepw:
            row['w'] = np.array(w, np.float64)
        self.rows.append(row)
        self.final[self._seed] = np.array(w, np.float64)
        if self.disp:
            fmt = {'float_kind': lambda v: '%.2e' % v}
            ms = ''.join('  %s %.3e' % (n, row[n]) for n in self.metrics or ())
            print('train %d%s  it %6d  t %7.2fs  run %3d  loss %.3e  terms %s'
                  '%s' % (self._ntrain, '' if self._seed is None else
                          ' seed %d' % self._seed, it, row['t'], run,
                          row['loss'], np.array2string(T, formatter=fmt), ms))
        return row

    def __getstate__(self):
        # plain data: no metric functions, model or arguments of a training
        state = dict(self.__dict__)
        if state.get('metrics'):
            state['metrics'] = {k: f if isinstance(f, str) else None
                                for k, f in state['metrics'].items()}
        state['_model'] = state['_args'] = None
        return state

    def _timeout(self):
        return self.tmax is not None and time.time() - self._t0 > self.tmax

    def _end(self, w, res):
        """End of a training: final weights and the result in 'info'
        (missing if the training was interrupted)."""
        self.final[self._seed] = np.array(w, np.float64)
        self.info[-1].update(time=time.time() - self._t0, nit=res.nit,
                             nouter=res.nouter, fun=float(res.fun),
                             status=res.status,
                             message=str(res.get('message', '')))
        self._model = self._args = None

    # --- results ---------------------------------------------------------

    @property
    def history(self):
        """The records as a dict of numpy arrays (one entry per row; 'terms'
        and 'lmbd' have one column per term, nan beyond the terms of a
        training with fewer terms than others)."""
        if not self.rows:
            return {}
        nt = max(len(r['terms']) for r in self.rows)

        def pad(v):  # fewer terms than the largest problem: nan columns
            return np.concatenate([v, np.full(nt - len(v), np.nan)])
        keys = list(dict.fromkeys(k for r in self.rows for k in r))
        h = {}
        for k in keys:
            if k in ('terms', 'lmbd'):
                h[k] = np.array([pad(r[k]) for r in self.rows])
            elif all(k in r for r in self.rows):
                h[k] = np.array([r[k] for r in self.rows])
            elif k == 'w':
                raise ValueError("weights kept only in some trainings "
                                 "(keepw)")
            else:  # a metric added later: nan where not recorded
                h[k] = np.array([r.get(k, np.nan) for r in self.rows])
        return h

    def plot(self, train=None, x='it', fname=None):
        """Terms, loss, lambdas and metrics of training 'train' (index,
        default the last one) against iterations (x='it') or time in seconds
        (x='t'); saved to 'fname' if given."""
        import matplotlib.pyplot as plt
        h = self.history
        if not h:
            raise ValueError("nothing recorded")
        train = h['train'][-1] if train is None else train
        s = h['train'] == train
        names = [k for k in h if k not in _COLUMNS]
        n = 3 + len(names)
        fig, ax = plt.subplots(n, 1, sharex=True, figsize=(7, 2.2*n))
        if x not in ('it', 't'):
            raise ValueError("x must be 'it' or 't'")
        it = h[x][s]
        cols = ~np.all(np.isnan(h['terms'][s]), axis=0)  # its own terms
        ax[0].semilogy(it, h['terms'][s][:, cols])
        ax[0].set_ylabel('terms')
        ax[0].legend(['term %d' % i for i in np.flatnonzero(cols)],
                     fontsize='small')
        ax[1].semilogy(it, h['loss'][s])
        ax[1].set_ylabel('loss')
        ax[2].semilogy(it, h['lmbd'][s][:, cols], drawstyle='steps-pre')
        ax[2].set_ylabel('lambdas')
        for a, name in zip(ax[3:], names):
            a.semilogy(it, np.abs(h[name][s]))
            a.set_ylabel(name)
        ends = it[h['end'][s]]
        for a in ax:
            for e in ends:
                a.axvline(e, color='grey', lw=0.5, ls=':')
        ax[-1].set_xlabel('iteration' if x == 'it' else 'time [s]')
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname)
        return fig

    def compare(self, key='loss', x='t', trains=None, term=None, ax=None,
                fname=None):
        """
        One curve per training: 'key' ('loss', a metric, or 'terms' or
        'lmbd' with the index 'term') against time in seconds (x='t') or
        iterations (x='it'), for the trainings 'trains' (indices, default
        all). The curves are labelled by what differs between the trainings
        (method, balance, seed). Drawn into 'ax' if given, saved to 'fname'
        if given; returns the figure.
        """
        import matplotlib.pyplot as plt
        h = self.history
        if not h:
            raise ValueError("nothing recorded")
        if key not in h:
            raise ValueError("no '%s' recorded, there are %s"
                             % (key, sorted(h)))
        if x not in ('it', 't'):
            raise ValueError("x must be 'it' or 't'")
        if h[key].ndim == 2:
            if term is None:
                raise ValueError("'%s' has one column per term, give 'term'"
                                 % key)
            y, ylabel = h[key][:, term], '%s %d' % (key, term)
        else:
            y, ylabel = h[key], key
        trains = sorted(set(h['train'].tolist())) if trains is None else \
            list(trains)
        info = {i['train']: i for i in self.info}
        fields = [f for f in ('method', 'balance', 'seed')
                  if len({str(info.get(i, {}).get(f)) for i in trains}) > 1]

        def label(i):
            parts = []
            for f in fields:
                v = info.get(i, {}).get(f)
                parts.append('seed %s' % v if f == 'seed' else str(v))
            return ', '.join(parts) if parts else 'train %d' % i
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3.5))
        else:
            fig = ax.figure
        for i in trains:
            s = h['train'] == i
            ax.semilogy(h[x][s], np.abs(y[s]), label=label(i))
        ax.set_xlabel('iteration' if x == 'it' else 'time [s]')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize='small')
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname)
        return fig
