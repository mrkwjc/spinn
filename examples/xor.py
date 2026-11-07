"""
Plain supervised training on data -- no physics involved.

This example shows that the 'spinn' class can be used directly as an
ordinary feed-forward neural network, fitted to input/target pairs only.
The default 'loss' method is just the mean squared error between network
outputs and targets, so no subclassing is needed:

    N = spinn(G, initx=x, inity=t)
    N.train(x, t)

Here the network learns the XOR logical function (output is 1 when
the inputs differ). The graph
mlgraph((2, 1, 1), full=True) has a single hidden neuron, but 'full=True'
also connects every layer to all later layers (inputs are wired directly
to the output). These shortcut connections make the problem solvable with
just one hidden neuron.

Passing 'initx' and 'inity' scales inputs and outputs to the data ranges.
"""
import jax.numpy as jnp
from spinn import spinn, mlgraph

x = jnp.asarray([[0., 0], [0, 1], [1, 0], [1, 1]])
t = jnp.asarray([[0.],    [1],    [1],    [0]])

G = mlgraph((2, 1, 1), full=True)
# G.preview()
N = spinn(G, initx=x, inity=t)
res = N.train(x, t, disp=True)

print(res)
print('Target: \n', t)
print('Output: \n', N(x))
