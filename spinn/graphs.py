import numpy as np
import networkx as nx
from spinn import utils


def _shoot(src, trg, mindeg, d=None, reverse=False):
    mindeg = min(mindeg, len(trg))
    if d is None:
        s = np.repeat(src, mindeg)
        t = [np.random.choice(trg, mindeg, replace=False)
             for i in range(len(src))]
    else:
        s, t = [], []
        for i, ei in enumerate(d):
            toshoot = mindeg-len(ei)
            if toshoot > 0:
                s += [src[i]]*toshoot
                if toshoot == mindeg:
                    ti = trg
                else:
                    ei = np.asarray(ei)
                    ni = ei[:, 0] if reverse else ei[:, 1]
                    ti = np.setdiff1d(trg, ni)
                t += [np.random.choice(ti, toshoot, replace=False)]
    if not len(s):
        return []
    e = np.asarray((s, np.concatenate(t)))
    e = e[::-1] if reverse else e
    return e.T.tolist()


def _multilayer(arch, full=False, mindeg=None, G=None):
    extents = utils.pacc((0,) + tuple(arch))
    layers = [range(start, end) for start, end in extents]
    G = nx.DiGraph() if G is None else G
    for i, layer in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    if not full:
        for layer1, layer2 in utils.pairwise(layers):
            if mindeg is None:
                G.add_edges_from(utils.prod(layer1, layer2))
            else:
                e = _shoot(layer1, layer2, mindeg)
                G.add_edges_from(e)
                d = [tuple(G.in_edges(n)) for n in layer2]
                e = _shoot(layer2, layer1, mindeg, d, reverse=True)
                G.add_edges_from(e)
    else:
        if mindeg is None:
            for i, layer1 in enumerate(layers[:-1]):
                for layer2 in layers[i+1:]:
                    G.add_edges_from(utils.prod(layer1, layer2))
        else:
            for i, layer1 in enumerate(layers[:-1]):
                trg = np.concatenate(layers[i+1:])
                e = _shoot(layer1, trg, mindeg)
                G.add_edges_from(e)
            for i, layer2 in enumerate(layers[::-1][:-1]):
                trg = np.concatenate(layers[::-1][i+1:])
                d = [tuple(G.in_edges(n)) for n in layer2]
                e = _shoot(layer2, trg, mindeg, d, reverse=True)
                G.add_edges_from(e)
    return G


class mlgraph(nx.DiGraph):
    def __init__(self, arch=None, full=False, d=None):
        nx.DiGraph.__init__(self)
        if arch is not None:
            _multilayer(arch, full, d, self)

    def preview(self, fname=None, **kwargs):
        pos = nx.multipartite_layout(self, subset_key="layer")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        nx.draw_networkx(self, pos, **kwargs)
        # plt.axis("equal")
        if fname is not None:
            plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    g = mlgraph((5, 20, 20, 1), full=True, d=3)
    # g.preview()
    g.preview(fname=None,
              with_labels=False,
              node_size=50,
              node_color='k',
              edge_color='grey',
              width=0.5,
              linewidths=0.5,
              arrows=False)
