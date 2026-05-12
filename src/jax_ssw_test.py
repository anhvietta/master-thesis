import jax
import jax.numpy as jnp
from jax import jit
from constants import blosum62_gttl, AMINO_ACIDS_GTTL
from ssw import sw_affine
from sw import sw_affine as swa
import matplotlib.pyplot as plt

# -----------------------
# Constants
# -----------------------
alphasize = len(AMINO_ACIDS_GTTL)

B = 1
l = 512
l1 = 400
l2 = 360

gap_ext = -1.0
gap_open = -11.0
temp = 1.0

key = jax.random.PRNGKey(0)

# -----------------------
# Mask construction
# -----------------------
mask = (
    (jnp.arange(l) < l1)[:, None]
    * (jnp.arange(l) < l2)[None, :]
)
mask = mask.astype(jnp.float32)
mask = jnp.expand_dims(mask, axis=0)
mask = jnp.broadcast_to(mask, (B, l, l))

# -----------------------
# Sequence generation
# -----------------------
key, k1, k2 = jax.random.split(key, 3)

s1 = jax.random.randint(
    k1,
    shape=(1, l1),
    minval=0,
    maxval=alphasize - 2,
)

s2 = jax.random.randint(
    k2,
    shape=(1, l2),
    minval=0,
    maxval=alphasize - 2,
)

# Pad with terminal symbol
s1 = jnp.pad(
    s1,
    pad_width=((0, 0), (0, l - l1)),
    constant_values=alphasize - 1,
)

s2 = jnp.pad(
    s2,
    pad_width=((0, 0), (0, l - l2)),
    constant_values=alphasize - 1,
)

# -----------------------
# Substitution matrix
# -----------------------
blosum62 = jnp.array(blosum62_gttl, dtype=jnp.float32)

# -----------------------
# Score matrix
# -----------------------
def scores(matrix, s1, s2):
    """
    s1: shape (L,) int32 or int64
    s2: shape (L,) int32 or int64
    returns: scores of shape (L, L)
    """
    s1 = s1.astype(jnp.int32)
    s2 = s2.astype(jnp.int32)
    L = l

    # Take rows corresponding to s1
    rows = jnp.take(matrix, s1, axis=0)  # shape (L, vocab_size)

    # Expand s2 to match rows for gather
    s2_exp = s2[None, :]  # shape (1, L)
    s2_exp = jnp.broadcast_to(s2_exp, (L, L))  # shape (L, L)

    # Gather values along last axis
    scores = jnp.take_along_axis(rows[:, None, :], s2_exp[:, :, None], axis=2)
    scores = scores.squeeze(-1)  # shape (L, L)
    return scores
scores_jax = jax.vmap(scores, in_axes=(None, 0, 0))
score_mats = scores_jax(blosum62, s1, s2)
score_mats = jnp.broadcast_to(score_mats, (B, l, l))

# -----------------------
# Smith-Waterman affine
# -----------------------
sw_fn = sw_affine(
    restrict_turns=True,
    penalize_turns=True,
    NINF=-1e-8,
    unroll=4,
    batch=True,
    eps=0
)

swa_fn = swa(
    restrict_turns=True,
    penalize_turns=True,
    NINF=-1e8,
    unroll=4,
    batch=True
)
lengths = jnp.broadcast_to(jnp.array([l1,l2], dtype=jnp.int32), (B,2))
result = sw_fn(score_mats, mask, gap_ext, gap_open, temp)
result_a = swa_fn(score_mats, lengths, gap_ext, gap_open, temp)
diff = result-result_a
print(result.min(), result.max(), score_mats.min(), score_mats.max())
print(result_a.min(), result_a.max())
print(diff.min(), diff.max(), diff.mean())
