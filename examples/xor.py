import jax.numpy as jnp
from spinn import spinn, mlgraph

x = jnp.asarray([[0., 0], [0, 1], [1, 0], [1, 1]])
t = jnp.asarray([[1.],    [0],    [0],    [1]])

G = mlgraph((2, 1, 1), full=True)
# G.preview()
N = spinn(G, initx=x, inity=t)
res = N.train(x, t, disp=True)

print(res)
print('Target: \n', t)
print('Output: \n', N(x))
