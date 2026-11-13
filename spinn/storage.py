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
