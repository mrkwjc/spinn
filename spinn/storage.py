"""
Files of spinn: numpy NPZ archives (zip of arrays) with the metadata as a
JSON text inside, under the name '__meta__'. Loading never unpickles, so
it does not run code from the file, and the content is readable with
numpy alone:

    z = np.load(fname)                       # the arrays
    meta = json.loads(str(z['__meta__']))    # the rest

'meta' always has 'format' (what the file holds, e.g. 'spinn.network'),
'version' (of that format) and 'spinn' (the version that wrote it).
"""
import json
import zipfile

import numpy as np

META = '__meta__'


def _plain(o):
    """JSON fallback: numpy numbers and arrays as Python values."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("cannot store %r (%s)" % (o, type(o).__name__))


def save(fname, fmt, version, meta, arrays=None):
    """Write 'arrays' (name: array) and 'meta' (JSON data) to 'fname' (the
    name is kept, numpy would add '.npz')."""
    from spinn import __version__
    meta = dict(meta, format=fmt, version=version, spinn=__version__)
    arrays = {k: np.asarray(v) for k, v in (arrays or {}).items()}
    if META in arrays:
        raise ValueError("'%s' is reserved" % META)
    with open(fname, 'wb') as f:
        np.savez(f, **arrays, **{META: np.array(json.dumps(
            meta, default=_plain))})


def isnpz(fname):
    """True if 'fname' is a zip archive (NPZ), e.g. not a pickle."""
    return zipfile.is_zipfile(fname)


def load(fname, fmt):
    """The arrays (dict) and metadata of an NPZ file of format 'fmt'."""
    with np.load(fname, allow_pickle=False) as z:
        if META not in z.files:
            raise ValueError("%s is not a spinn file (no metadata)" % fname)
        meta = json.loads(str(z[META]))
        arrays = {k: z[k] for k in z.files if k != META}
    if meta.get('format') != fmt:
        raise ValueError("%s holds '%s', not '%s'"
                         % (fname, meta.get('format'), fmt))
    return arrays, meta


def tuples(x):
    """Lists (from JSON) as tuples again, nested."""
    if isinstance(x, list):
        return tuple(tuples(v) for v in x)
    return x


def path(obj):
    """Importable name 'module:qualname' of a class or function, None if
    it has none (a lambda, a local function or class)."""
    mod = getattr(obj, '__module__', None)
    qual = getattr(obj, '__qualname__', None)
    if not mod or not qual or '<' in qual:
        return None
    return '%s:%s' % (mod, qual)


def importname(name):
    """The class or function of a name from 'path' (imports its module)."""
    import importlib
    mod, qual = name.split(':')
    obj = importlib.import_module(mod)
    for part in qual.split('.'):
        obj = getattr(obj, part)
    return obj


def _isarray(v):
    return isinstance(v, np.ndarray) or (
        hasattr(v, '__array__') and hasattr(v, 'shape') and
        hasattr(v, 'dtype') and not isinstance(v, np.generic))


def encode(v, arrays, objects=True):
    """'v' as JSON data; arrays go to 'arrays' (name: array) and are
    referred to by name. Tuples, dicts, classes, functions (by importable
    name; others become None) and objects (class name and attributes) are
    marked so that 'decode' restores them; objects=False refuses classes,
    functions and objects (TypeError)."""
    if v is None or isinstance(v, (bool, str, int, float)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if _isarray(v):
        name = 'a%d' % len(arrays)
        arrays[name] = np.asarray(v)
        return {'@array': name}
    if isinstance(v, tuple):
        return {'@tuple': [encode(x, arrays, objects) for x in v]}
    if isinstance(v, list):
        return [encode(x, arrays, objects) for x in v]
    if isinstance(v, dict):
        if all(isinstance(k, str) for k in v):
            return {'@dict': {k: encode(x, arrays, objects)
                              for k, x in v.items()}}
        return {'@items': [[encode(k, arrays, objects),
                            encode(x, arrays, objects)]
                           for k, x in v.items()]}
    if not objects:
        raise TypeError("cannot store %s as plain data" % type(v).__name__)
    if isinstance(v, type):
        return {'@class': path(v)}
    if callable(v) and not hasattr(v, '__dict__') or \
            type(v).__name__ in ('function', 'builtin_function_or_method'):
        return {'@func': path(v), 'name': getattr(v, '__name__', repr(v))}
    if hasattr(v, '__dict__'):
        cls = path(type(v))
        if cls is None:
            raise TypeError("cannot store an instance of the local class %s"
                            % type(v).__qualname__)
        return {'@object': cls, 'attrs': encode(vars(v), arrays, objects)}
    raise TypeError("cannot store %r (%s)" % (v, type(v).__name__))


def decode(v, arrays):
    """Inverse of 'encode' (imports the modules of classes, functions and
    objects; a function without importable name gives None)."""
    if isinstance(v, list):
        return [decode(x, arrays) for x in v]
    if not isinstance(v, dict):
        return v
    if '@array' in v:
        return arrays[v['@array']]
    if '@tuple' in v:
        return tuple(decode(x, arrays) for x in v['@tuple'])
    if '@dict' in v:
        return {k: decode(x, arrays) for k, x in v['@dict'].items()}
    if '@items' in v:
        return {decode(k, arrays): decode(x, arrays) for k, x in v['@items']}
    if '@class' in v:
        return importname(v['@class'])
    if '@func' in v:
        return None if v['@func'] is None else importname(v['@func'])
    if '@object' in v:
        cls = importname(v['@object'])
        obj = cls.__new__(cls)
        obj.__dict__.update(decode(v['attrs'], arrays))
        return obj
    raise ValueError("unknown entry in the file: %r" % (v,))
