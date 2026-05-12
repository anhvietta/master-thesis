from typing import Optional, Sequence
from sw import sw_affine
import jax
import jax.numpy as jnp
from jax.nn import relu, softmax, sigmoid
from flax import linen as nn
from flax import nnx
from constants import max_length, attention_mask_window_size, blosum62_tensor, blosum62_std, blosum62_gttl
from utils import tokens, pad_token

dtype = jnp.bfloat16
param_dtype = jnp.float32

def make_padmask(q_padmask, d_padmask):
    """
    q_padmask: (Q,)
    d_padmask: (D,)
    returns:   (Q, D)
    """
    return jnp.expand_dims(q_padmask, 1) | jnp.expand_dims(d_padmask, 0)

def compute_sim_mat(q_emb, d_emb, q_padmask, d_padmask):
    """
    q_emb:      (Q, D)
    d_emb:      (K, D)
    q_padmask:  (Q,)
    d_padmask:  (K,)
    """
    sim = jnp.einsum("qd,kd->qk", q_emb, d_emb)
    sim_padmask = make_padmask(q_padmask, d_padmask)
    sim = jnp.where(sim_padmask, -1.0, sim)
    return sim, sim_padmask

def normalize(x, ord=2, axis=-1, eps=1e-6):
    norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    return x / (norm + eps)

def sim_loss_fn(weight, target, normalized=True):
    if normalized:
        weight = normalize(weight)
    aa_sim = weight @ weight.T
    return -jnp.mean(aa_sim * target)

def mmatch(sim, matchmask, nummatch, temp, abs_scale=0.1):
    scaled_sim = sim / temp
    mask = jnp.sum(matchmask, axis=1)

    pos_logits = jnp.sum(scaled_sim * matchmask, axis=1)
    lse = jax.scipy.special.logsumexp(scaled_sim, axis=1)

    l = -(pos_logits - lse) * mask
    l = jnp.sum(l) / nummatch

    aux_loss = jnp.sum(jnp.square((1 - sim) * matchmask)) / nummatch
    return l + abs_scale * aux_loss

def match_loss(mats, mask, temp=0.1, abs_scale=0.1):
    nummatch = jnp.sum(mask) + 1e-4

    qmatch = mmatch(mats, mask, nummatch, temp, abs_scale)
    dmatch = mmatch(mats.T, mask.T, nummatch, temp, abs_scale)

    return (qmatch + dmatch) / 2.0

def smatch(sim, matchmask, nummatch):
    selection = jnp.sum(relu(sim) * matchmask, axis=1)
    denom = jnp.sum(relu(sim), axis=1) + 1e-4
    loss = selection / denom
    loss = jnp.sum(loss) / nummatch
    return 1.0 - loss

def signal_loss(mats, mask):
    nummatch = jnp.sum(mask) + 1e-4

    qmatch = smatch(mats, mask, nummatch)
    dmatch = smatch(mats.T, mask.T, nummatch)

    return (qmatch + dmatch) / 2.0

def get_expected_idx(m):
    probs = softmax(m, axis=1)
    positions = jnp.arange(max_length)
    return jnp.sum(probs * positions, axis=1)

def make_seq_softmask(span, length, temp=0.2, padding=0):
    # span: [2] → (start, end)

    start, end = span[0], span[1]

    xs = jnp.linspace(0.0, length - 1.0, length)

    half1 = sigmoid((xs - start) / temp)
    half2 = sigmoid((end - xs) / temp)

    return half1 * half2

def locality_loss(indices, span, temp=0.2, padding=0):
    # indices: [L]
    # span: [2]

    # Equivalent to F.relu(indices[:, :-1] - indices[:, 1:])
    indices_diff = relu(indices[:-1] - indices[1:])     # [L-1]

    mask = make_seq_softmask(
        span,
        length=max_length - 1,
        temp=temp,
        padding=padding,
    )

    return jnp.mean(indices_diff * mask)

def traceback_loss_fn(traceback, mask, eps=1e-8):
    #num_match = jnp.sum(mask) + eps
    traceback = jnp.clip(traceback, eps, 1-eps)
    ce = -jnp.mean(mask * jnp.log(traceback) + (1-mask) * jnp.log(1-traceback))
    #pos = -jnp.mean(jnp.log(traceback * mask + eps) * mask)
    #neg = -jnp.mean(jnp.log())
    return ce

def loss_fn(
    mats, #[L, L]
    traceback, #[L, L]
    spanref, #[2, 2]
    seqmatchmask, #[2, L]
    seqspanmask, #[2, L]
    pairmatchmask, #[L, L]
    temp=0.08,
    abs_scale=0.1,
    max_scale=1,
):
    coord_loss = match_loss(mats, pairmatchmask, temp, abs_scale)
    #traceback_loss = match_loss(traceback, pairmatchmask, temp=temp, abs_scale=0.0)
    traceback_loss = traceback_loss_fn(traceback, pairmatchmask)

    q_idx = get_expected_idx(mats)
    d_idx = get_expected_idx(mats.T)

    q_order_loss = locality_loss(q_idx, spanref[0, :])
    d_order_loss = locality_loss(d_idx, spanref[1, :])

    order_loss = q_order_loss + d_order_loss

    loss = coord_loss + traceback_loss

    return (
        loss,
        order_loss,
        coord_loss,
        traceback_loss,
        signal_loss(mats, pairmatchmask)
    )

class Tokenizer(nn.Module):
    vocab_size: int
    init_token: Optional[jnp.ndarray] = None
    normalized: bool = True
    pad_idx: int = tokens.index(pad_token)  # pass tokens.index(pad_token) here

    def setup(self):
        if self.init_token is not None:
            init_token = self.preprocess_token(self.init_token, self.vocab_size)
            token_dim = init_token.shape[1]

            self.embedding = nn.Embed(
                num_embeddings=self.vocab_size,
                features=token_dim,
                embedding_init=lambda *_: init_token
            )
            self.layer_norm = nn.LayerNorm(
                dtype=dtype,
                param_dtype=param_dtype
            )

            self.outdim = token_dim
        else:
            token_dim = 30
            self.embedding = nn.Embed(
                num_embeddings=self.vocab_size,
                features=token_dim
            )
            self.layer_norm = nn.LayerNorm(
                dtype=dtype,
                param_dtype=param_dtype
            )

            self.outdim = token_dim

    def preprocess_token(self, init_token, vocab_size):
        init_token = jnp.asarray(init_token, dtype=jnp.float32)

        if init_token.shape[0] < vocab_size:
            pad_len = vocab_size - init_token.shape[0]
            init_token = jnp.pad(
                init_token,
                ((0, pad_len), (0, 0)),
                mode="constant"
            )
        elif init_token.shape[0] > vocab_size:
            raise RuntimeError("Incompatible init tokens")

        assert init_token.shape[0] == vocab_size
        return init_token

    def __call__(self, input_ids):
        # input_ids: (batch, seq_len)
        tokenized = self.embedding(input_ids.astype(jnp.int32))
        tokenized = self.layer_norm(tokenized)

        if self.normalized:
            norm = jnp.linalg.norm(tokenized, ord=2, axis=-1, keepdims=True)
            tokenized = tokenized / (norm + 1e-12)

        return tokenized

class ConvBlock(nn.Module):
    kernel_size: int
    in_dim: int
    out_dim: int
    dilation: int
    groups: int

    """@nn.compact
    def __call__(self, x, *, train: bool):
        # x: (L, C)
        x = nn.Conv(
            features=self.out_dim,
            kernel_size=(self.kernel_size,),
            kernel_dilation=(self.dilation,),
            padding="SAME",
            feature_group_count=self.groups,
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=nn.initializers.zeros
        )(x)
        x = nn.gelu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x"""

    @nn.compact
    def __call__(self, x, *, train: bool):
        # x: (L, C)
        x = nn.Conv(
            features=self.in_dim,
            kernel_size=(self.kernel_size,),
            kernel_dilation=(self.dilation,),
            padding="SAME",
            feature_group_count=self.groups,
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=nn.initializers.zeros,
            dtype=dtype,
            param_dtype=param_dtype
        )(x)
        x = nn.GroupNorm(num_groups=self.in_dim // 16, use_bias=True, use_scale=True)(x)
        x = nn.gelu(x)
        x = nn.Conv(
            features=self.out_dim,
            kernel_size=(1,),
            kernel_dilation=(1,),
            padding="SAME",
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=nn.initializers.zeros,
            dtype=dtype,
            param_dtype=param_dtype
        )(x)
        x = nn.LayerNorm(
            dtype=dtype,
            param_dtype=param_dtype
        )(x)
        return x

class ConvolutionalEncoder(nn.Module):
    token_dim: int
    dim: Sequence[int]
    conv_kernel: Sequence[int]
    dilation: Sequence[int]

    def setup(self):
        init_dim = self.dim[0]

        self.ffn = [
            nn.Dense(
                init_dim * 2,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                dtype=dtype,
                param_dtype=param_dtype
            ),
            nn.gelu,
            nn.LayerNorm(
                init_dim * 2,
                dtype=dtype,
                param_dtype=param_dtype
            ),
            nn.Dense(
                init_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                dtype=dtype,
                param_dtype=param_dtype
            ),
            nn.gelu,
        ]

        self.conv_blocks = [
            ConvBlock(
                kernel_size=self.conv_kernel[i],
                in_dim=self.dim[i],
                out_dim=self.dim[i],
                dilation=self.dilation[i],
                groups=init_dim,
            )
            if i == 0
            else ConvBlock(
                kernel_size=self.conv_kernel[i],
                in_dim=self.dim[i - 1],
                out_dim=self.dim[i],
                dilation=self.dilation[i],
                groups=init_dim,
            )
            for i in range(len(self.dim))
        ]
    
    def __call__(self, tokenized, *, train: bool):
        """
        tokenized: (K, L, token_dim)
        returns:   (K, L, dim[-1])
        """

        # FFN
        x = tokenized
        for layer in self.ffn:
            x = layer(x)

        # Conv expects (L, C), so vmap over K
        def apply_conv(seq):
            # seq: (L, C)
            z = seq
            for block in self.conv_blocks:
                z = block(z, train=train)
            return z

        x = jax.vmap(apply_conv)(x)
        return x

    def encode(self, tokenized, *, train: bool):
        return self(tokenized, train=train)

class ProtConvColBERT(nn.Module):
    token_dim: int = 24
    dim: Sequence[int] = (32,)
    conv_kernel: Sequence[int] = (5,)
    dilation: Sequence[int] = (1,)
    outdim: int = 512

    def setup(self):
        assert (
            len(self.dim) == len(self.conv_kernel) == len(self.dilation) > 0
        )

        self.encoder = ConvolutionalEncoder(
            token_dim=self.token_dim,
            dim=self.dim,
            conv_kernel=self.conv_kernel,
            dilation=self.dilation,
        )

        projection_dim = self.dim[-1]

        self.proj_dense1 = nn.Dense(
            2 * projection_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=dtype,
            param_dtype=param_dtype
        )
        self.norm1 = nnx.RMSNorm(
            2 * projection_dim,
            rngs=nnx.Rngs(0),
            dtype=dtype,
            param_dtype=param_dtype
        )
        self.proj_dense2 = nn.Dense(
            self.outdim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=dtype,
            param_dtype=param_dtype
        )
        self.norm2 = nnx.RMSNorm(
            self.outdim,
            rngs=nnx.Rngs(0),
            dtype=dtype,
            param_dtype=param_dtype
        )

    def __call__(self, tokenized, *, train: bool):
        """
        tokenized: (K, L, token_dim)
        returns:   (K, L, outdim)
        """
        x = self.encoder(tokenized, train=train)
        x = self.proj_dense1(x)
        x = self.norm1(x)
        x = nn.relu(x)
        x = self.proj_dense2(x)
        x = self.norm2(x)
        x = normalize(x)

        return x

    def encode(self, seq, *, train: bool):
        x = self.encoder.encode(seq, train=train)
        x = self.proj_dense1(x)
        x = nn.relu(x)
        x = self.proj_dense2(x)
        return normalize(x)

SSW_MATRIX_MAX = 1

class SSW(nn.Module):
    matrix: jnp.ndarray
    unroll: int
    gap_open: float
    gap_ext: float
    restrict_turns:bool
    penalize_turns:bool
    NINF:float=-1e30
    temp:float=1.0
    eps:float=1e-8

    def setup(self):
        self.ssw_func = sw_affine(
            batch=False,
            restrict_turns=self.restrict_turns,
            penalize_turns=self.penalize_turns,
            NINF=self.NINF,
            unroll=self.unroll,
            eps=0
        )
        self.alpha = self.param(
            "alpha",
            nn.initializers.constant(0.1),
            ()
        )
        '''self.beta = self.param(
            "beta",
            nn.initializers.constant(0.),
            ()
        )'''

    def __call__(self, seq, lengths, mats):
        score_mats = self.similarity_to_score_transform(mats, seq)
        traceback = self.ssw_func(
            score_mats.astype(jnp.float32),
            lengths,
            self.gap_ext * self.alpha,
            self.gap_open * self.alpha,
            self.temp
        )
        return traceback

    def similarity_to_score_transform(self, sim_mat, seq):
        '''scores = self.scores(seq[0,:], seq[1,:])
        scored_mats = sim_mat * scores
        return scored_mats'''
        return sim_mat

    def scores(self, s1, s2):
        """
        s1: shape (L,) int32 or int64
        s2: shape (L,) int32 or int64
        returns: scores of shape (L, L)
        """
        s1 = s1.astype(jnp.uint8)
        s2 = s2.astype(jnp.uint8)
        L = s1.shape[0]

        # Take rows corresponding to s1
        rows = jnp.take(self.matrix, s1, axis=0)  # shape (L, vocab_size)

        # Expand s2 to match rows for gather
        s2_exp = s2[None, :]  # shape (1, L)
        s2_exp = jnp.broadcast_to(s2_exp, (L, L))  # shape (L, L)

        # Gather values along last axis
        scores = jnp.take_along_axis(rows[:, None, :], s2_exp[:, :, None], axis=2)
        scores = scores.squeeze(-1)  # shape (L, L)
        return scores

class JaxEncoder(nn.Module):
    dims: Sequence[int]
    conv_kernels: Sequence[int]
    dilations: Sequence[int]
    outdim: int
    ssw_matrix: Optional[jnp.ndarray]
    ssw_unroll: int
    gap_open: float
    gap_ext: float
    ssw_NINF: float
    ssw_temp: float
    ssw_restrict_turns: bool
    ssw_penalize_turns: bool
    ssw_eps: float
    vocab_size: int = len(tokens)
    init_token: Optional[jnp.ndarray] = None
    guide_blosum: Optional[jnp.ndarray] = None
    normalized: bool = True


    def setup(self):
        assert (
            len(self.dims)
            == len(self.conv_kernels)
            == len(self.dilations)
        )

        self.tokenizer = Tokenizer(
            vocab_size=self.vocab_size,
            init_token=self.init_token,
            normalized=self.normalized,
        )

        token_dim = self.tokenizer.outdim

        # single encoder (refactored)
        self.encoder = ProtConvColBERT(
            token_dim=token_dim,
            dim=self.dims,
            conv_kernel=self.conv_kernels,
            dilation=self.dilations,
            outdim=self.outdim,
        )

        self.ssw = SSW(
            matrix=self.ssw_matrix,
            restrict_turns=self.ssw_restrict_turns,
            penalize_turns=self.ssw_penalize_turns,
            unroll=self.ssw_unroll,
            NINF=self.ssw_NINF,
            temp=self.ssw_temp,
            eps=self.ssw_eps,
            gap_open=self.gap_open,
            gap_ext=self.gap_ext
        )

    def get_traceback(self, seq, lengths, mats):
        return self.ssw(seq, lengths, mats)

    def get_dummy(self, seq, lengths, mats):
        return jnp.zeros_like(mats, dtype=jnp.float32)

    def __call__(self, seq, seq_padmask, lengths, *, with_traceback:bool=False, train:bool=False):
        """
        seq:          (2, L)
        seq_padmask:  (2, L)
        """
        tokenized = self.tokenizer(seq)
        seqemb = self.encoder(tokenized, train=train)

        mats, pair_padmask = compute_sim_mat(
            seqemb[0, :],
            seqemb[1, :],
            seq_padmask[0, :],
            seq_padmask[1, :]
        )

        traceback = self.get_traceback(
            seq,
            lengths,
            mats
        )

        return mats, traceback

    def with_tb(self, seq, seq_padmask, lengths, *, train:bool=False):
        """
        seq:          (2, L)
        seq_padmask:  (2, L)
        """
        tokenized = self.tokenizer(seq)
        seqemb = self.encoder(tokenized, train=train)

        mats, pair_padmask = compute_sim_mat(
            seqemb[0, :],
            seqemb[1, :],
            seq_padmask[0, :],
            seq_padmask[1, :]
        )

        traceback = self.get_traceback(
            seq,
            lengths,
            mats
        )

        return mats, traceback

    def encode(self, seq, *, train:bool=False):
        tokenized = self.tokenizer(seq)
        return self.encoder.encode(tokenized, train)
