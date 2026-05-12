import os
import torch
import torch.nn as nn
import torch.nn.functional as F
#from x_transformers import Encoder
from constants import max_length, attention_mask_window_size, blosum62_tensor, blosum62_std, blosum62_gttl
from utils import tokens, pad_token
import math
import random
from torch.amp import autocast
from typing import Optional
from torch.utils.checkpoint import checkpoint
'''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax2torch import jax2torch
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "gpu")
jax.devices("gpu")
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(0.5))
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, D)
        return self.scalar * x + self.pe[:, :, :]

class SymmetricConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=0, groups=1):
        super().__init__()
        half_size = (kernel_size + 1) // 2
        self.weight_half = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, half_size)
        )
        self.kernel_size = kernel_size
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.padding=padding
        self.groups=groups
        nn.init.kaiming_uniform_(self.weight_half, a=5**0.5)

    def forward(self, x):
        # Mirror the half kernel
        if self.kernel_size % 2 == 0:
            mirrored = torch.cat([self.weight_half,
                                  torch.flip(self.weight_half, dims=[-1])],
                                 dim=-1)
        else:
            mirrored = torch.cat([self.weight_half,
                                  torch.flip(self.weight_half[:, :, :-1], dims=[-1])],
                                 dim=-1)

        return F.conv1d(x, mirrored, bias=self.bias, padding=self.padding, groups=self.groups)

class RelativePositionalEncoding(nn.Module):
    def __init__(self, maxlen, d_model, max_relative_distance):
        super().__init__()
        self.max_relative_distance = max_relative_distance
        self.rel_embeddings = nn.Embedding(2 * max_relative_distance + 1, d_model)
        pos = torch.arange(maxlen)
        rel = pos[None, :] - pos[:,None]
        rel = torch.clamp(rel, -max_relative_distance, max_relative_distance)
        rel = rel + max_relative_distance
        self.register_buffer("rel_position_indices", rel, persistent=False)

    def forward(self, x):
        B, L, D = x.shape
        rel_emb = self. rel_embeddings(self.rel_position_indices[:L,:L])
        rel_bias = rel_emb.mean(dim=1)
        return x + rel_bias.unsqueeze(0)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, maxlen, d_model):
        super().__init__()
        self.positional_embeddings = nn.Embedding(maxlen, d_model)
        nn.init.uniform_(self.positional_embeddings.weight, -0.1, 0.1)

    def forward(self, x):
        L = x.size(1)
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        positional_encoding = self.positional_embeddings(positions)
        return x + positional_encoding

class Tokenizer(nn.Module):
    def __init__(self, vocab_size, init_token: Optional[torch.Tensor]=None, normalized=True):
        super().__init__()
        self.normalized = normalized
        if init_token is not None:
            init_token = self.preprocess_token(init_token, vocab_size)
            token_dim = init_token.shape[1]
            self.embedding = nn.Sequential(
                nn.Embedding.from_pretrained(init_token, freeze=False, padding_idx=tokens.index(pad_token)),
            )
            self.outdim = token_dim
        else:
            token_dim = 30
            self.embedding = nn.Sequential(
                nn.Embedding(vocab_size, token_dim, padding_idx=tokens.index(pad_token)),
            )
            self.outdim = token_dim

    def preprocess_token(self, init_token, vocab_size):
        init_token = torch.tensor(init_token, dtype=torch.float)
        if init_token.shape[0] < vocab_size:
            init_token = F.pad(init_token, (0,0,0,vocab_size-init_token.shape[0]), 'constant', 0)
        elif init_token.shape[0] > vocab_size:
            raise RuntimeError("Incompatible init tokens")
        assert(init_token.shape[0] == vocab_size)
        return init_token

    def forward(self, input_ids):
        tokenized = self.embedding(input_ids.int())
        if self.normalized:
            tokenized = F.normalize(tokenized, p=2., dim=-1)
        return tokenized

    def get_embedding(self):
        return self.embedding[0].weight

class StackingEncoder(nn.Module):
    def __init__(self, token_dim=24, dim=[32], conv_kernel=[9], dilation=[1], symmetric=True):
        super().__init__()
        assert all([k % 2 == 1 for k in conv_kernel]), "Kernel size should be odd for symmetric windowing."
        self.symmetric=symmetric
        self.kernel_size = conv_kernel
        init_dim = dim[0]
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, init_dim*2),
            nn.GELU(),
            nn.Linear(init_dim*2, init_dim),
            nn.GELU(),
        )
        if symmetric:
            init_kernel = torch.normal(mean=1., std=1, size=(conv_kernel // 2 + 1,))
            init_kernel = init_kernel / (2 * init_kernel.sum() - init_kernel[-1])
        else:
            init_kernel = torch.normal(mean=1., std=1, size=(conv_kernel,))
            init_kernel = init_kernel / (init_kernel.sum())
        self.kernel = nn.Parameter(init_kernel)

    def forward(self, tokenized):
        x = self.ffn(tokenized)
        padding = self.kernel_size // 2
        B, K, L, Dtok = x.shape
        #x = F.normalize(x, p=2., dim=-1)
        x = self.ffn(x.view(B*K,L,Dtok))
        Dtok = x.size(2)
        x = x.view(B,K,L,Dtok)
        x_padded = F.pad(x, (0,0,padding,padding)) #B, K, L, D
        x_stacked = x_padded.unfold(dimension=2,size=self.kernel_size,step=1) #B, K, L, D, kernel_size
        if self.symmetric:
            mirrored = torch.cat([self.kernel, torch.flip(self.kernel[:-1], dims=[-1])], dim=-1) # kernel_size
            kernel = mirrored.expand(Dtok, self.kernel_size)
        else:
            kernel = self.kernel.expand(Dtok, self.kernel_size)
        emb = (x_stacked * kernel).transpose(3,4).reshape(B, K, L, Dtok * self.kernel_size)
        return F.normalize(emb, p=2., dim=-1)

    def encode(self,tokenized):
        x = self.ffn(tokenized)
        z = self.conv(x.transpose(1, 2)).transpose(2, 1)  # Apply conv (B, L, D)
        z = self.feedforward(z)
        return z

class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, kernel_size, dilation, groups, symmetric, skip=True, ptwise=True, last=False):
        super().__init__()
        conv_class = SymmetricConv1d if symmetric else nn.Conv1d
        self.skip = skip
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            conv_class(indim, outdim if not ptwise else indim, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=groups),  # depthwise
            nn.BatchNorm1d(outdim if not ptwise else indim)
        )
        if not last or ptwise:
            self.conv.append(nn.GELU())
        if ptwise:
            self.conv.extend(nn.Sequential(
                nn.Conv1d(indim, outdim, kernel_size=1),  # ptwise
                nn.BatchNorm1d(outdim)
            ))
            if not last:
                self.conv.append(nn.GELU())

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.conv(x)
        if self.skip:
            z += x
        return z.transpose(1,2)

class ConvolutionalEncoder(nn.Module):
    def __init__(self, token_dim, dim, conv_kernel=[9], dilation=[1], symmetric=False, init_token: Optional[torch.Tensor]=None, normalized=True, skip=True, ptwise=True, extract=[0]):
        super().__init__()
        self.dim = dim
        self.n_layers = len(dim)
        self.normalized = normalized
        self.extract = extract
        init_dim = dim[0]

        self.ffn = nn.Sequential(
            nn.Linear(token_dim, init_dim*2),
            nn.GELU(),
            nn.Linear(init_dim*2, init_dim),
            nn.GELU(),
        )
        self.convs = nn.ModuleList()
        self.convs.append(self.make_conv_block(dim[0], dim[0], conv_kernel[0], dilation[0], init_dim, symmetric, skip, ptwise))
        for i in range(self.n_layers-1):
            in_dim = dim[i]
            out_dim = dim[i+1]
            conv = self.make_conv_block(in_dim, out_dim, conv_kernel[i+1], dilation[i+1], init_dim, symmetric, skip, ptwise, last=True)
            self.convs.append(conv)

        '''ffn_dim = dim[-1]

        self.feedforward = nn.Sequential(
            nn.Linear(ffn_dim, ffn_dim * 2),
            nn.GELU(),
            nn.Linear(ffn_dim * 2, ffn_dim),
            nn.LayerNorm(ffn_dim),
            nn.GELU()
        )'''

    def forward(self, tokenized):
        x = self.ffn(tokenized)
        B, K, L, Dtok = x.shape
        x = x.view(B * K, L, Dtok)
        out = []
        for i, conv in enumerate(self.convs):
            x = conv(x)  # Apply conv (B, L, D)
            if i in self.extract:
                _, _, Demb = x.shape
                out.append(x.view(B, K, L, Demb)) 
        return out  # shape: (B, L, dim)

    def encode(self,tokenized):
        x = self.ffn(tokenized)
        out = []
        for i, conv in enumerate(self.convs):
            x = conv(x)  # Apply conv (B, L, D)
            if i in self.extract:
                out.append(x)
        return out

    def make_conv_block(self, indim, outdim, kernel_size, dilation, groups, symmetric, skip, ptwise, last=False):
        return ConvBlock(
            indim, outdim, kernel_size=kernel_size, dilation=dilation, groups=groups, symmetric=symmetric, skip=skip, ptwise=ptwise, last=last
        )

class ConvolutionalPositionalEncoder(nn.Module):
    def __init__(self, dim, conv_kernel=9, maxlen=max_length, symmetric=True):
        super().__init__()
        assert conv_kernel % 2 == 1, "Kernel size should be odd for symmetric windowing."
        self.symmetric=symmetric
        self.embedding = nn.Parameter(torch.empty((dim, conv_kernel), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.embedding)
        self.kernel_size = conv_kernel
        self.conv_filter = nn.Conv1d(dim, dim, conv_kernel, padding=0, groups=dim)

    def forward(self, x):
        padding = self.kernel_size // 2
        B, L, Dtok = x.shape
        x = x.view(B,L,Dtok)
        x_padded = F.pad(x, (0, 0, padding, padding)) #B, L, D
        x_stacked = x_padded.unfold(dimension=1, size=self.kernel_size, step=1) #B, L, D, kernel_size
        x_stacked += self.embedding[None, None, :, :]
        emb = self.conv_filter(x_stacked.reshape(B*L, Dtok, self.kernel_size))
        #return F.relu(x_stacked.mean(dim=-1))
        return emb.reshape(B, L, Dtok)

class StackingConvolutionalEncoder(nn.Module):
    def __init__(self, token_dim, dim, conv_kernel=[9], dilation=[1], symmetric=False):
        super().__init__()
        self.dim = dim
        self.n_layers = len(dim)
        self.normalized = normalized
        self.pos_enc = ConvolutionalPositionalEncoder(
            maxlen=max_length,
            dim=dim,
            conv_kernel=conv_kernel
        )

        self.ffn = nn.Sequential(
            nn.Linear(token_dim, init_dim*2),
            nn.GELU(),
            nn.Linear(init_dim*2, init_dim),
            nn.GELU(),
        )
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, tokenized):
        x = self.ffn(tokenized)
        B, K, L, Dtok = x.shape
        x = x.view(B * K, L, Dtok)
        #padding_mask = padding_mask.view(B * K, L)
        z = self.pos_enc(x)
        z = self.feedforward(z)
        #z = z + x
        _, _, Demb = z.shape
        z = z.view(B, K, L, Demb)
        #padding_mask = padding_mask.view(B, K, L)
        return z  # shape: (B, L, dim)

    def encode(self,tokenized):
        x = self.ffn(tokenized)
        z = self.conv(x.transpose(1, 2)).transpose(2, 1)  # Apply conv (B, L, D)
        z = self.feedforward(z)
        return z

class ProtConvColBERT(nn.Module):
    def __init__(
        self,
        token_dim=24,
        dim=[32],
        conv_kernel=[5],
        dilation=[1],
        vocab_size=len(tokens),
        enc_type=0,
        symmetric=False,
        outdim=512,
        skip=True,
        ptwise=True,
        extract=[0]
    ):
        super().__init__()
        assert(len(dim) == len(conv_kernel) > 0 and len(dim) == len(conv_kernel) and len(dim) == len(dilation))
        if enc_type == 0:
            self.encoder = ConvolutionalEncoder(
                token_dim=token_dim,
                dim=dim,
                dilation=dilation,
                conv_kernel=conv_kernel,
                symmetric=symmetric,
                skip=skip,
                ptwise=ptwise,
                extract=extract
            )
            projection_dim = dim[-1]
        elif enc_type == 1:
            self.encoder = StackingConvolutionalEncoder(
                token_dim=token_dim,
                dim=dim,
                dilation=dilation,
                conv_kernel=conv_kernel,
                symmetric=symmetric
            )
            projection_dim = dim[-1]
        else:
            self.encoder = StackingEncoder(
                dim=dim,
                token_dim=token_dim,
                dilation=dilation,
                conv_kernel=conv_kernel,
                max_len=max_length,
                symmetric=symmetric
            )
            projection_dim = dim[-1]
        extract_sel = [i in extract for i in range(len(dim))]
        self.projection_head = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.RMSNorm(2 * d),
                nn.ReLU(inplace=True),
                nn.Linear(2 * d, outdim),
                nn.RMSNorm(outdim)
            ) for d, e in zip(dim, extract_sel) if e
        )
        self.scale = nn.Parameter(torch.ones((len(self.projection_head))))

    def forward(self, tokenized):
        x = self.encoder(tokenized)  # (B, 2, L, dim)
        out = []
        for i, ph in enumerate(self.projection_head):
            o = x[i]
            out.append(ph(o))
        out = torch.stack(out, dim=1)
        weights = torch.softmax(self.scale, dim=0)
        weights = weights.view(1, -1, 1, 1, 1)
        out = (out * weights).sum(dim=1)
        out = F.normalize(out, p=2., dim=-1, eps=1e-6)
        return out

    def encode(self, seq):
        x = self.encoder.encode(seq)
        out = []
        for i, ph in enumerate(self.projection_head):
            o = x[i]
            out.append(ph(o))
        out = torch.stack(out, dim=1)
        weights = torch.softmax(self.scale, dim=0)
        weights = weights.view(1, -1, 1, 1)
        out = (out * weights).sum(dim=1)
        out = F.normalize(out, p=2., dim=-1, eps=1e-6)
        return out

    @staticmethod
    def make_padmask(q_padmask, d_padmask):
        return q_padmask.unsqueeze(2) | d_padmask.unsqueeze(1)

    def compute_sim_mat(self, q_emb, d_emb, q_padmask, d_padmask):
        '''if torch.isnan(q_emb).any() or torch.isnan(d_emb).any():
            raise ValueError("NaN detected in embeddings!")
        if torch.isinf(q_emb).any() or torch.isinf(d_emb).any():
            raise ValueError("Inf detected in embeddings!")'''
        sim = torch.einsum('bqd,bkd->bqk', q_emb, d_emb)
        sim_padmask = ProtConvColBERT.make_padmask(q_padmask, d_padmask)
        sim = sim.masked_fill(sim_padmask, -1)
        return sim

    @staticmethod
    def make_seq_mask(batch_size, span, length=max_length):
        mask = torch.zeros(batch_size, length)
        for i in range(batch_size):
            mask[i, span[0][i]: span[1][i]] = 1
        return mask

    @staticmethod
    def make_seq_softmask(batch_size, span, length=max_length, temp=0.2, padding=0):
        start = span[:, 0]# * (length - 1)
        end = span[:, 1]# * (length - 1)

        # Create pixel grid
        xs = torch.linspace(0, length - 1, length, device=span.device).repeat((span.size(0),1))

        half1 = torch.sigmoid((xs - start.unsqueeze(1))/temp)
        half2 = torch.sigmoid((end.unsqueeze(1) - xs)/temp)
        # Compute sigmoid-based edges
        mask = half1 * half2
        return mask

    @staticmethod
    def colbert_score(query_emb, doc_emb, qspan, dspan):
        batch_size, Lq, D = query_emb.shape
        _, Ld, _ = doc_emb.shape
        device = query_emb.get_device()
        query_mask = ProtConvColBERT.make_seq_mask(batch_size, qspan).detach().to(device) if qspan else None
        doc_mask = ProtConvColBERT.make_seq_mask(batch_size, dspan).detach().to(device) if dspan else None
        sim = torch.einsum('bqd,bkd->bqk', query_emb, doc_emb)
        #sim = torch.clamp(sim, min=-50, max=50)
        if doc_mask is not None:
            sim = sim.masked_fill(doc_mask.unsqueeze(1) == 0, -1e4)
        max_sim, _ = sim.max(dim=2)
        probs = F.softmax(sim, dim=2)         # shape: (B, Q, D)
        positions = torch.arange(D).to(sim.device)
        expected_idx = (probs * positions).sum(dim=2)  # shape: (B, Q)
        if query_mask is not None:
            max_sim = max_sim * query_mask
            expected_idx = expected_idx * query_mask
        return max_sim.sum(dim=1), expected_idx

    @staticmethod
    def singular_locality_loss(indices, span, temp=0.2, padding=0):
        indices_diff = F.relu(indices[:, :-1] - indices[:, 1:])
        mask = ProtConvColBERT.make_seq_softmask(indices_diff.size(0), span, length=max_length-1, temp=temp, padding=padding)
        locality_loss = (indices_diff * mask).mean()
        return locality_loss

    @staticmethod
    def kendall_locality_loss(indices):
        locality_penalty = (indices[:, None, :] - indices[:, :, None])  # shape: (B, Q, Q)
        mask = torch.triu(torch.ones_like(locality_penalty), diagonal=1)
        violations = (locality_penalty < 0).float() * mask
        locality_loss = violations.sum()
        return locality_loss

    @staticmethod
    def reference_locality_loss(expected_idx, span, ref_idx):
        pass

    @staticmethod
    def loss_fn(query_emb, pos_emb, neg_emb, qpspan, pspan, qnspan, nspan, use_span, margin=0.5, locality_coef=0.0001):
        query_emb = F.normalize(query_emb, p=2., dim=-1)
        pos_emb   = F.normalize(pos_emb, p=2., dim=-1)
        neg_emb   = F.normalize(neg_emb, p=2., dim=-1)

        if use_span:
            if random.random() < 0.5:
                ps = qpspan
                pos_score, pos_idx = ProtConvColBERT.colbert_score(query_emb, pos_emb, ps, None)  # (B,)
            else:
                ps = pspan
                pos_score, pos_idx = ProtConvColBERT.colbert_score(pos_emb, query_emb, ps, None)
            if random.random() < 0.5:
                ns = qnspan
                neg_score, neg_idx = ProtConvColBERT.colbert_score(query_emb, neg_emb, ns, None)  # (B,)
            else:
                ns = nspan
                neg_score, neg_idx = ProtConvColBERT.colbert_score(neg_emb, query_emb, ns, None)
        else:
            pos_score, pos_idx = ProtConvColBERT.colbert_score(query_emb, pos_emb, None, None)  # (B,)
            neg_score, neg_idx = ProtConvColBERT.colbert_score(query_emb, neg_emb, None, None)  # (B,)

        # Compute ranking loss
        target = torch.ones_like(pos_score)
        ranking_loss = F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)

        # Locality loss
        locality_loss = ProtConvColBERT.singular_locality_loss(pos_idx, ps) + ProtConvColBERT.singular_locality_loss(neg_idx, ns)

        loss = ranking_loss + locality_coef * locality_loss

        return loss, ranking_loss, locality_loss

class BlosumScorer(nn.Module):
    def __init__(self, blosum_matrix):
        super().__init__()
        self.matrix = torch.tensor(blosum_matrix, dtype=torch.float32)
        #self.conv_kernel = torch.eye(conv_kernel).unsqueeze(0).unsqueeze(0) / conv_kernel
        self.padding = conv_kernel // 2
        kernel = torch.ones(conv_kernel, dtype=torch.float32)
        kernel = kernel / conv_kernel
        self.conv_kernel = nn.Parameter(kernel)

    def scores(self, s1, s2):
        #rows = self.blosum[s1.reshape(-1)].reshape(s1.size(0), s1.size(1), -1)
        B, L = s1.shape
        rows = torch.index_select(self.matrix, 0, s1.reshape(-1)).reshape(B, L, -1)
        s2_exp = s2.unsqueeze(1).expand(-1, s1.size(1), -1).to(dtype=torch.int64)
        scores = torch.gather(rows, 2, s2_exp)
        return scores

    def forward(self, input_ids):
        assert(input_ids.size(1) in [2,3])
        input_ids = input_ids.int()
        q = input_ids[:,0,:]
        p = input_ids[:,1,:]

        B, L = q.shape
        qp = self.scores(q,p)

        if input_ids.size(1) == 3:
            n = input_ids[:,2,:]
            qn = self.scores(q,n)
            scores = torch.stack((qp,qn), dim=1).view(B*2, 1, L, L)
        else:
            scores = qp.unsqueeze(1)

        diagonalized_kernel = torch.diag_embed(self.conv_kernel).unsqueeze(0).unsqueeze(0)
        conv_scores = F.conv2d(scores, diagonalized_kernel.to(device=scores.device, dtype=scores.dtype), padding=self.padding)

        return conv_scores.view(B, input_ids.size(1)-1, L, L)

def sw_affine(restrict_turns=True,
             penalize_turns=True,
             batch=True, unroll=2, NINF=-1e30, temp=1.0):
    """smith-waterman (local alignment) with affine gap"""
    # rotate matrix for vectorized dynamic-programming
    def rotate(x):
        # solution from jake vanderplas (thanks!)
        a,b = x.shape
        dtype = x.dtype
        ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
        i,j = (br-ar)+(a-1),(ar+br)//2
        n,m = (a+b-1),(a+b)//2
        output = {
            "x":jnp.full([n,m],NINF, dtype=x.dtype).at[i,j].set(x),
            "o":(jnp.arange(n)+a%2)%2
        }
        return output, (
            jnp.full((m,3), NINF, dtype=x.dtype),
            jnp.full((m,3), NINF, dtype=x.dtype)
        ), (i,j)

    # fill the scoring matrix
    def sco(x, mask, gap=0.0, open=0.0):
        dtype = x.dtype
        NINF_array = jnp.array(NINF, dtype=x.dtype)

        def _soft_maximum(x, axis=None):
            def _logsumexp(y):
                y = jnp.maximum(y,NINF_array)
                return jax.nn.logsumexp(y, axis=axis)
            return temp*_logsumexp(x/temp)

        def sm2(x, mask, axis=None):
            def _logsumexp(y):
                y = jnp.maximum(y,NINF_array)
                tmp = jnp.exp(y - y.max(axis, keepdims=True))
                return y.max(axis) + jnp.log(jnp.sum(mask * tmp, axis=axis))
            return temp*_logsumexp(x/temp)


        def _cond(cond, true, false): return cond*true + (1-cond)*false
        def _pad(x,shape): return jnp.pad(x,shape,constant_values=(NINF,NINF))

        def _step(prev, sm):
            h2,h1 = prev   # previous two rows of scoring (hij) mtxs

            Align = jnp.pad(h2,[[0,0],[0,1]]) + sm["x"][:,None]
            Right = _cond(sm["o"], _pad(h1[:-1],([1,0],[0,0])),h1)
            Down  = _cond(sm["o"], h1,_pad(h1[1:],([0,1],[0,0])))

            # add gap penalty
            if penalize_turns:
                Right += jnp.stack([open,gap,open])
                Down += jnp.stack([open,open,gap])
            else:
                gap_pen = jnp.stack([open,gap,gap])
                Right += gap_pen
                Down += gap_pen

            if restrict_turns: Right = Right[:,:2]

            h0_Align = _soft_maximum(Align,-1)
            h0_Right = _soft_maximum(Right,-1)
            h0_Down = _soft_maximum(Down,-1)
            h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
            return (h1,h0),h0

        sm, prev, idx = rotate(x[:-1,:-1])
        hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]

        # sink
        return sm2(hij + x[1:,1:,None], mask=mask[1:,1:,None])
    # traceback to get alignment (aka. get marginals)
    traceback = jax.grad(sco)

    # add batch dimension
    if batch: return jax.vmap(traceback,(0,0,None,None))
    else: return traceback

CHUNK_SIZE = 40

class Soft_SW_JAX(nn.Module):
    def __init__(self, score_mat=blosum62_gttl, gap_open=-11., gap_ext=-1., temp=0.7, NINF=-1e4):
        super().__init__()
        self.matrix = torch.abs(torch.tensor(score_mat, dtype=torch.float16))
        self.gap_open = torch.tensor(gap_open,dtype=torch.float16)
        self.gap_ext = torch.tensor(gap_ext,dtype=torch.float16)
        self.temp = torch.tensor(temp,dtype=torch.float16)
        self.ssw_func = jax2torch(jax.jit(
            sw_affine(
                restrict_turns=False,
                penalize_turns=False,
                NINF=NINF,
                temp=temp
            )
        ))
        self.ckpt_ssw = lambda x, mask: checkpoint(
            self.ssw_func,
            x,
            mask,
            self.gap_ext,
            self.gap_open,
            use_reentrant=False
        )
        self.NINF=NINF

    @torch.compiler.disable
    def forward(self, seq, padmask, mats):
        assert(mats.grad_fn is not None or not torch.is_grad_enabled())
        self.matrix = self.matrix.to(device=seq.device)
        self.gap_open = self.gap_open.to(device=seq.device)
        self.gap_ext = self.gap_ext.to(device=seq.device)
        self.temp = self.temp.to(device=seq.device)

        scores = self.scores(seq[:,0,:], seq[:,1,:])
        scored_mats = mats[:,0,:,:] * scores
        traceback = []
        B = scored_mats.size(0)
        #device = scored_mats.device
        '''padded_sm = F.pad(
            scored_mats,
            pad=(1,0,1,0),
            value=self.NINF
        )
        padded_mask = F.pad(
            padmask,
            pad=(1,0,1,0),
            value=1
        )'''
        #print(padded_sm.max(), padded_sm.min())
        #scored_mats_cpu = scored_mats.cpu()
        num_chunks = B // CHUNK_SIZE + (0 if scored_mats.size(0) % CHUNK_SIZE == 0 else 1)
        half_mask = (~padmask).half()
        for i in range(num_chunks):
            s = i * CHUNK_SIZE
            e = min(s + CHUNK_SIZE, B)
            traceback.append(
                self.run_ssw(
                    scored_mats[s:e,:,:],
                    half_mask[s:e,:,:]
                )
            )
        return torch.cat(traceback, dim=0)

    def run_ssw(self, x, mask):
        if torch.is_grad_enabled():
            return self.ckpt_ssw(x, mask)
        else:
            return self.ssw_func(x, mask, self.gap_ext, self.gap_open)

    def scores(self, s1, s2):
        B, L = s1.shape
        s1 = s1.to(dtype=torch.int64)
        s2 = s2.to(dtype=torch.int64)
        rows = torch.index_select(self.matrix, 0, s1.reshape(-1)).reshape(B, L, -1)
        s2_exp = s2.unsqueeze(1).expand(-1, s1.size(1), -1)
        scores = torch.gather(rows, 2, s2_exp)
        return scores

class Soft_SW(nn.Module):
    def __init__(self, score_mat=blosum62_gttl, gap_open=11., gap_ext=1., temp=1., NINF=-1e4):
        super().__init__()
        self.matrix = torch.abs(torch.tensor(score_mat, dtype=torch.float32))
        self.matrix = (self.matrix - self.matrix.min())/(self.matrix.max() - self.matrix.min())
        self.gap_open = gap_open
        self.gap_ext = gap_ext
        self.temp = nn.Parameter(torch.tensor(temp))
        self.NINF = NINF

    def forward(self, seq, padmask, mats):
        self.matrix = self.matrix.to(device=seq.device)
        scores = self.scores(seq[:,0,:], seq[:,1,:])
        scored_mats = mats[:,0,:,:] * scores
        padded_sm = F.pad(
            scored_mats,
            pad=(1,0,1,0),
            value=self.NINF
        )
        padded_mask = F.pad(
            padmask,
            pad=(1,0,1,0),
            value=1
        )
        assert(padded_sm.grad_fn is not None)
        score = self.sco(padded_sm, padded_mask, 0.)
        #print(score.shape)
        return self.traceback(score, scored_mats)

    @torch.compiler.disable
    def traceback(self, score, scored_mats):
        return torch.autograd.grad(
            score,
            scored_mats,
            create_graph=True
        )[0]

    @torch.compile
    def batch_rotate(self, x):
        """
        Rotate DP matrix for vectorized anti-diagonal processing
        """
        B, _, _ = x.shape
        L = max_length
        device = x.device

        ar = torch.arange(L - 1, -1, -1, device=device)[:, None]
        br = torch.arange(L, device=device)[None, :]

        i = (br - ar) + (L - 1)
        j = (ar + br) // 2

        n = L * 2 - 1
        m = L

        out = torch.full((B, n, m), self.NINF, device=device)
        batch_idx = torch.arange(B, device=device)[:, None, None]  # (B, 1, 1)
        #out[batch_idx, i[None], j[None]] = x
        #mask = torch.zeros_like(out, dtype=torch.bool)
        #mask[batch_idx, i[None], j[None]] = True
        #out = torch.where(mask, x, out)
        linear_idx = i * m + j          # (L, L)
        linear_idx = linear_idx.view(1, -1).expand(B, -1)  # (B, L*L)

        out_flat = out.view(B, -1)      # (B, n*m)
        src = x.reshape(B, -1)          # (B, L*L)

        out_flat = out_flat.scatter(
            dim=1,
            index=linear_idx,
            src=src
        )
        out = out_flat.view(B, n, m)

        assert(out.grad_fn is not None)
        o = (torch.arange(n, device=device) + (L % 2)) % 2

        prev = (
            torch.full((B, m, 3), self.NINF, device=device),
            torch.full((B, m, 3), self.NINF, device=device),
        )

        return {"x": out, "o": o}, prev, (i, j)

    @torch.compile
    def sco(self, scored_mat, mask, eps):
        def soft_maximum(x, dim=-1, mask=None):
            def lse(y):
                y = torch.maximum(y, torch.tensor(self.NINF, device=x.device))
                if mask is None:
                    return torch.logsumexp(y, dim=dim)
                else:
                    ymax = y.max()
                    return ymax + torch.log(torch.sum(~mask[..., None] * torch.exp(y - ymax)) + eps)
            return self.temp * lse(x/self.temp)

        def cond(c, t, f):
            return c * t + (1 - c) * f

        def pad(x, pad_shape):
            return F.pad(x, pad_shape, value=self.NINF)

        x = scored_mat.masked_fill(mask, self.NINF)
        gap_open = self.gap_open
        gap_ext = self.gap_ext

        sm, prev, idx = self.batch_rotate(x[:,:-1,:-1])
        h2, h1 = prev
        gap_pen = torch.tensor([gap_open, gap_ext, gap_ext], device=x.device)

        outputs = []

        for t in range(sm["x"].shape[1]):
            sm_t = sm["x"][:,t,:]
            o_t = sm["o"][t]

            Align = F.pad(h2, (0, 1)) + sm_t.unsqueeze(-1)

            Right = cond(
                o_t,
                pad(h1[:, :-1, :], (0, 0, 1, 0)),
                h1
            ) + gap_pen

            Down = cond(
                o_t,
                h1,
                pad(h1[:, 1:, :], (0, 0, 0, 1))
            ) + gap_pen

            h0_align = soft_maximum(Align, dim=-1)
            h0_right = soft_maximum(Right, dim=-1)
            h0_down  = soft_maximum(Down, dim=-1)

            h0 = torch.stack([h0_align, h0_right, h0_down], dim=-1)

            h2, h1 = h1, h0
            outputs.append(h0)

        i, j = idx
        batch_idx = torch.arange(x.size(0), device=x.device)[:, None, None]
        hij = torch.stack(outputs, dim=1)[batch_idx, i[None], j[None], :]
        assert(hij.grad_fn is not None)
        score = soft_maximum(
            hij + x[:, 1:, 1:, None],
            mask=mask[:, 1:, 1:]
        )
        return score

    def scores(self, s1, s2):
        B, L = s1.shape
        s1 = s1.to(dtype=torch.int64)
        s2 = s2.to(dtype=torch.int64)
        rows = torch.index_select(self.matrix, 0, s1.reshape(-1)).reshape(B, L, -1)
        s2_exp = s2.unsqueeze(1).expand(-1, s1.size(1), -1)
        scores = torch.gather(rows, 2, s2_exp)
        return scores

def blosum_to_distance(blosum):
    """
    Convert BLOSUM similarity matrix to a distance-like matrix.
    """
    blosum = blosum.float()
    blosum = (blosum - blosum.mean()) / blosum.std()
    dist = blosum.max() - blosum
    dist.fill_diagonal_(0.0)
    return dist


def knn_heat_kernel(dist, k=5, sigma=None):
    """
    Build symmetric kNN graph with heat kernel weights.
    """

    n = dist.shape[0]

    if sigma is None:
        sigma = torch.median(dist)

    W = torch.zeros_like(dist)

    for i in range(n):
        d = dist[i]
        idx = torch.argsort(d)

        neighbors = idx[1:k+1]

        weights = torch.exp(-(d[neighbors] ** 2) / (2 * sigma ** 2))

        W[i, neighbors] = weights

    # symmetrize graph
    W = 0.5 * (W + W.T)

    return W


def graph_laplacian(W, normalized=True):
    """
    Compute graph Laplacian.
    """
    deg = W.sum(dim=1)

    if normalized:
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-8))
        L = torch.eye(W.shape[0], device=W.device) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        D = torch.diag(deg)
        L = D - W

    return L


class AminoAcidLaplacianRegularizer(nn.Module):

    def __init__(self, blosum62, k=5, sigma=None, normalized=True):
        super().__init__()

        blosum = torch.tensor(blosum62, dtype=torch.float32)

        dist = blosum_to_distance(blosum)

        W = knn_heat_kernel(dist, k=k, sigma=sigma)

        L = graph_laplacian(W, normalized=normalized)

        self.register_buffer("laplacian", L)

    def forward(self, embeddings):
        """
        embeddings: (20, d)
        """

        L = self.laplacian.to(device=embeddings.device)

        loss = torch.trace(embeddings.T @ L @ embeddings)

        return loss

class ColBERT_direct(nn.Module):
    def __init__(self,
            dims=[32],
            conv_kernels=[5],
            dilations=[1],
            vocab_size=len(tokens),
            enc_type=0,
            symmetric=False,
            init_token: Optional[torch.Tensor]=None,
            guide_blosum: Optional[torch.Tensor]=None,
            knn: int=5,
            sigma: Optional[float]=None,
            outdims=[512],
            normalized=True,
            skip=True,
            ptwise=True,
            extracts=[0]
        ):
        super().__init__()
        assert(len(dims) == len(dilations) and len(dims) == len(conv_kernels) and len(dims) == len(outdims))
        self.tokenizer = Tokenizer(
            vocab_size=vocab_size,
            init_token=init_token,
            normalized=normalized
        )
        token_dim = self.tokenizer.outdim

        self.colbert = nn.ModuleList(
            ProtConvColBERT(
                dim=dim,
                token_dim=token_dim,
                conv_kernel=conv_kernel,
                dilation=dilation,
                enc_type=enc_type,
                symmetric=symmetric,
                outdim=outdim,
                skip=skip,
                ptwise=ptwise,
                extract=extract
            ) for dim, conv_kernel, dilation, outdim, extract in zip(dims, conv_kernels, dilations, outdims, extracts)
        )
        self.num_enc = len(dims)
        self.simloss_scale = nn.Parameter(torch.tensor(5.0))
        self.guide_blosum = guide_blosum
        self.outdims = outdims
        self.normalized=normalized
        self.regularizer = AminoAcidLaplacianRegularizer(
            blosum62=guide_blosum,
            k=knn,
            sigma=sigma
        )
        #self.ssw = Soft_SW_JAX(temp=1.)

    def forward(self, seq, seq_padmask, num_enc_used):
        tokenized = self.tokenizer(seq)
        '''seqemb = self.colbert[0](tokenized)
        mats, pair_padmask = self.colbert[0].compute_sim_mat(seqemb[:,0,:,:], seqemb[:,1,:,:], seq_padmask[:,0,:], seq_padmask[:,1,:])
        mats = mats.unsqueeze(1)
        #traceback = self.ssw(seq, pair_padmask, mats)
        #print(traceback.min(), traceback.max())
        #assert(not torch.isnan(traceback).any())
        #assert(traceback.grad_fn is not None or not torch.is_grad_enabled())
        return mats, traceback'''
        mats = []
        for m in self.colbert:
            seqemb = m(tokenized)
            mats.append(m.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,1,:,:], seq_padmask[:,0,:], seq_padmask[:,1,:]))
        return torch.stack(mats,dim=1)

    @torch.jit.export
    def encode(self, seq):
        return self.colbert[0].encode(self.tokenizer(seq))
    
    def sub_sim_loss(self, weight, target):
        if self.normalized:
            normalized = F.normalize(weight, p=2., dim=-1)
        else:
            normalized = weight
        aa_sim = normalized @ normalized.T           # (24,24)
        return - (aa_sim * target).mean()

    def sim_loss(self, device):
        if self.guide_blosum is None:
            return torch.tensor(0.0)
        target = self.guide_blosum.to(device)
        return self.sub_sim_loss(self.tokenizer.embedding[0].weight, target)
        #return self.regularizer(self.tokenizer.embedding[0].weight)

    def mmatch(self, sim, matchmask, nummatch, temp, abs_scale=0.1, margin=0.6):
        scaled_sim = sim / temp
        mask = matchmask.sum(dim=2)
        pos_logits = (scaled_sim * matchmask).sum(dim=2)
        lse = torch.logsumexp(scaled_sim, dim=2)
        l = -(pos_logits - lse) * mask
        l = l.sum(dim=1)/nummatch
        #aux_loss = torch.square((1-sim) * matchmask).sum(dim=(1,2))/nummatch
        aux_loss = F.relu((margin - sim) * matchmask).sum(dim=(1,2))/nummatch
        return l + abs_scale * aux_loss

    def match_loss(self, mats, pair_match_mask, temp = 0.1, abs_scale=0.1):
        with autocast("cuda", enabled=False):
            B, C, L, _ = mats.shape
            assert(mats.shape[3] == L and C == 1)
            mat_reshaped = mats.view((B*C,L,L)).float()
            pair_match_mask_reshaped = pair_match_mask.view((B*C, L, L))
            nummatch = pair_match_mask_reshaped.sum(dim=(1,2)) + 1e-4
            qmatch = self.mmatch(mat_reshaped, pair_match_mask_reshaped, nummatch, temp, abs_scale)
            dmatch = self.mmatch(mat_reshaped.transpose(1,2), pair_match_mask_reshaped.transpose(1,2), nummatch, temp, abs_scale)
            m_loss = (qmatch/2 + dmatch/2).mean(dim=0)
            return m_loss

    def smatch(self, sim, matchmask, nummatch):
        selection = (F.relu(sim) * matchmask).sum(dim=2)
        denom = F.relu(sim).sum(dim=2) + 1e-4
        loss = selection/denom
        loss = loss.sum(dim=1)/nummatch
        return 1-loss

    def signal_loss(self, mats, pair_match_mask):
        B, C, L, _ = mats.shape
        assert(mats.shape[3] == L and C == 1)
        mat_reshaped = mats.view((B*C,L,L))
        pair_match_mask_reshaped = pair_match_mask.view((B*C, L, L))
        nummatch = pair_match_mask_reshaped.sum(dim=(1,2))
        qmatch = self.smatch(mat_reshaped, pair_match_mask_reshaped, nummatch)
        dmatch = self.smatch(mat_reshaped.transpose(1,2), pair_match_mask_reshaped.transpose(1,2), nummatch)
        s_loss = (qmatch/2 + dmatch/2).mean(dim=0)
        return s_loss

    def get_expected_idx(self, m):
        probs = F.softmax(m, dim=2)
        positions = torch.arange(max_length).to(m.device)
        return (probs * positions).sum(dim=2)

    def match_loss_get(self, mats, pair_match_mask, temp = 0.1, abs_scale=0.1, num_enc_used=1, threshold=0.4):
        assert(mats.shape[1] >= num_enc_used)
        #print(mats.shape)
        loss = torch.tensor(0.0, device=mats.device)
        refmask = pair_match_mask.float().clone()
        for i in range(num_enc_used):
            loss += self.match_loss(mats[:,i,:,:,:], refmask, temp=temp, abs_scale=abs_scale)
            refmask = (torch.abs(refmask - mats[:,i,:,:,:]) > (1-threshold)).float() * refmask
        return loss

    @torch.compiler.disable
    def get_traceback_loss(self, traceback, pairmatchmask, eps=1e-8):
        normalized = traceback / traceback.sum(dim=(1,2), keepdim=True) + eps
        return -(torch.log(traceback) * pairmatchmask).sum(dim=(1,2)).mean()

    #@torch.compiler.disable
    def loss_fn(self, mats, spanref, seqmatchmask, seqspanmask, pairmatchmask, temp=0.08, mask_padding=0, guide_scale=0.001, abs_scale=0.1, max_scale=0.5, num_enc_used=1, threshold=0.4):
        mats = mats.unsqueeze(2)
        mats_reduced = temp*torch.logsumexp(mats/temp, dim=1, keepdim=True)
        coord_loss = self.match_loss_get(mats, pairmatchmask, temp=temp, abs_scale=abs_scale, num_enc_used=num_enc_used, threshold=threshold)
        max_coord_loss = self.match_loss_get(mats_reduced, pairmatchmask, temp=temp, abs_scale=abs_scale, num_enc_used=1, threshold=threshold)
        #traceback_loss = self.get_traceback_loss(traceback, pairmatchmask)
        #traceback_loss = self.match_loss(traceback.unsqueeze(1), pairmatchmask, temp=0.1, abs_scale=0)
        mats_reduced = mats_reduced.squeeze(1)
        q_idx = self.get_expected_idx(mats_reduced[:,0,:,:])
        d_idx = self.get_expected_idx(mats_reduced[:,0,:,:].transpose(1,2))
        q_order_loss = ProtConvColBERT.singular_locality_loss(q_idx, spanref[:,0,:], temp=0.08, padding=mask_padding)
        d_order_loss = ProtConvColBERT.singular_locality_loss(d_idx, spanref[:,1,:], temp=0.08, padding=mask_padding)
        order_loss = q_order_loss + d_order_loss
        sloss = self.sim_loss(device=mats.device)
        loss = (coord_loss + max_scale * max_coord_loss)/(num_enc_used+max_scale)  + guide_scale * sloss
        return loss, order_loss, coord_loss, max_coord_loss, self.signal_loss(mats_reduced, pairmatchmask), sloss
