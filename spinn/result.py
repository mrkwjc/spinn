"""
Result of 'spinn.train' and 'spinn.resume', saved and loaded as one file.

    res = P.train(x, bx, monitor=100)
    res.save('beam.res')                 # network, result, monitor, state
    res.save('beam.res', data=True)      # also the data of the training

    res = result.load('beam.res')        # all: res.model, res.state
    res = result.load('beam.res', model=False)   # result and monitor only

A result is a scipy OptimizeResult (a dict with attribute access) with
'save' and 'load' added. 'res.model' gives the network of the result: a
new copy at the weights 'res.x' on each access, so 'res' never changes
(keep a copy to work with it: P2 = res.model).

The file is a numpy NPZ archive (see spinn.storage): readable with numpy
alone, and loading runs no code from the file. It holds the network
(weights, structure and the name of its class), the result fields, the
monitor (as plain data: metric functions by name only) and the state for
'resume' (balancer and optimizer with their state; the data of the
training only with data=True). load(model=False) needs nothing but spinn:
it gives 'res.network', the network as a plain spinn (to evaluate, not to
train), and no state. The full load imports the class of the network and
those of its balancer and optimizer by their names (a class defined in a
script run as __main__ is found only from that script); functions which
have no importable name (lambdas, local functions, e.g. the schedules of
optax) are not stored and become None.
"""
import copy

from scipy.optimize import OptimizeResult

__all__ = ['result']

FORMAT, VERSION = 'spinn.result', 1
_HIDDEN = ('_model', '_cls')               # not shown
_OWN = ('_model', '_cls', 'state', 'monitor', 'network')   # stored apart


class result(OptimizeResult):
    __doc__ = __doc__

    def __repr__(self):
        return repr(OptimizeResult({k: v for k, v in self.items()
                                    if k not in _HIDDEN}))

    def __dir__(self):
        return [k for k in super().__dir__() if k not in _HIDDEN]

    def __getattr__(self, name):
        if name == 'model':  # the property found no model
            raise AttributeError(
                "no model: the result was loaded with model=False (see "
                "'network') or comes from an optimizer, not from train")
        return super().__getattr__(name)

    @property
    def model(self):
        """The network of the result (a new copy at the weights 'x')."""
        m = self.get('_model')
        if m is None:
            raise AttributeError('model')  # message in __getattr__
        import jax.numpy as jnp
        c = copy.copy(m)
        c.w = jnp.asarray(self['x'], jnp.asarray(m.w).dtype)
        return c

    def _net(self):
        """The network to store (model at 'x', or the loaded 'network')."""
        if self.get('_model') is not None:
            return self.model
        if self.get('network') is not None:
            return self['network']
        raise ValueError("the result has no network to save")

    def save(self, fname, data=False):
        """Save to 'fname' (NPZ, see 'result'); data=True also stores the
        data of the training (else 'resume' needs them as arguments)."""
        from spinn import storage
        arrays, skipped = {}, []
        fields = {}
        for k, v in self.items():
            if k in _OWN:
                continue
            try:
                fields[k] = storage.encode(v, arrays, objects=False)
            except TypeError:
                skipped.append(k)   # e.g. scipy's hess_inv
        net = self._net()
        nmeta, narr = net._plain()
        arrays.update({'net_' + k: v for k, v in narr.items()})
        cls = self['_model'] if self.get('_model') is not None else None
        meta = dict(fields=fields, skipped=skipped, network=nmeta,
                    cls=storage.path(type(cls)) if cls is not None
                    else self.get('_cls'))
        mon = self.get('monitor')
        meta['monitor'] = None if mon is None else mon._plain(arrays)
        st = self.get('state')
        meta['state'] = None if st is None else _state_plain(st, arrays,
                                                             data)
        storage.save(fname, FORMAT, VERSION, meta, arrays)

    @classmethod
    def load(cls, fname, model=True):
        """Result saved by 'save'. model=True: also the network as its own
        class ('model') and the state for 'resume' (imports their classes);
        model=False: only result fields, monitor and 'network' (a plain
        spinn), needs nothing but spinn."""
        from spinn import storage
        from spinn.monitor import monitor
        from spinn.trees import spinn
        arrays, meta = storage.load(fname, FORMAT)
        res = cls({k: storage.decode(v, arrays)
                   for k, v in meta['fields'].items()})
        narr = {k[4:]: v for k, v in arrays.items() if k.startswith('net_')}
        if meta['monitor'] is not None:
            res['monitor'] = monitor._from_plain(meta['monitor'], arrays)
        if not model:
            res['network'] = spinn._from_plain(meta['network'], narr)
            res['_cls'] = meta['cls']
            return res
        try:
            mcls = storage.importname(meta['cls'])
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(
                "the class of the network, %s, cannot be imported (%s); "
                "load(fname, model=False) gives the results and the network "
                "as a plain spinn" % (meta['cls'], e)) from e
        res['_model'] = mcls._from_plain(meta['network'], narr)
        if meta['state'] is not None:
            st = _state_from_plain(meta['state'], arrays)
            st.monitor = res.get('monitor')
            res['state'] = st
        return res


_STATE = ('opt', 'b', 'method', 'balance', 'maxiter', 'seed', 'lmbd', 'x',
          'nit', 'nfev', 'k', 'short', 'phase', 'last', 'pending', 'lastrun',
          'resume', 'mtrain')


def _state_plain(st, arrays, data):
    from spinn import storage
    d = {k: getattr(st, k) for k in _STATE}
    if data:
        d['args'] = st.args
    return storage.encode(d, arrays)


def _state_from_plain(plain, arrays):
    from spinn import storage
    from spinn.training import trainer
    d = storage.decode(plain, arrays)
    st = trainer.__new__(trainer)
    st.__dict__.update(d)
    st.args = d.get('args')     # None: not stored, 'resume' needs them
    st.monitor = None
    return st
