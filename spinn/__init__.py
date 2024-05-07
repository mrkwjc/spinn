import jax
jax.config.update('jax_default_matmul_precision', 'high')  # 'bfloat16_3x'

from .trees import spinn
from .utils import hcube, left, right, bottom, top
from .graphs import mlgraph
