import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder
from constants import max_length, attention_mask_window_size, blosum62_tensor, blosum62_std, blosum62_gttl
from utils import tokens, pad_token
import math
import random
from typing import Optional

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

class StackingEncoder(nn.Module):
    def __init__(self, vocab_size, dim=[32], conv_kernel=[9], dilation=[1], symmetric=True, init_token: Optional[torch.Tensor]=None):
        super().__init__()
        assert all([k % 2 == 1 for k in conv_kernel]), "Kernel size should be odd for symmetric windowing."
        self.symmetric=symmetric
        if init_token is not None:
            init_token = self.preprocess_token(init_token, vocab_size)
            token_dim = init_token.shape[1]
            self.embedding = nn.Sequential(
                nn.Embedding.from_pretrained(init_token, freeze=False, padding_idx=tokens.index(pad_token)),
                nn.LayerNorm((max_len, token_dim))
            )
        else:
            token_dim = dim[0]
            self.embedding = nn.Sequential(
                nn.Embedding(vocab_size, dim[0], padding_idx=tokens.index(pad_token)),
                nn.LayerNorm((max_len, token_dim))
            )
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, dim[0]),
            nn.GELU(),
        )
        self.kernel_size = conv_kernel
        if symmetric:
            init_kernel = torch.normal(mean=1., std=1, size=(conv_kernel // 2 + 1,))
            init_kernel = init_kernel / (2 * init_kernel.sum() - init_kernel[-1])
        else:
            init_kernel = torch.normal(mean=1., std=1, size=(conv_kernel,))
            init_kernel = init_kernel / (init_kernel.sum())
        self.kernel = nn.Parameter(init_kernel)

    def preprocess_token(self, init_token, vocab_size):
        init_token = torch.tensor(init_token, dtype=torch.float)
        if init_token.shape[0] < vocab_size:
            init_token = F.pad(init_token, (0,0,0,vocab_size-init_token.shape[0]), 'constant', 0)
        elif init_token.shape[0] > vocab_size:
            raise RuntimeError("Incompatible init tokens")
        assert(init_token.shape[0] == vocab_size)
        return init_token

    def forward(self, input_ids):
        padding = self.kernel_size // 2
        x = self.embedding(input_ids.int()) #B, K, L, D
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

    def encode(self,input_ids):
        tokenized = self.embedding(input_ids.int())
        if self.normalized:
            tokenized = F.normalize(tokenized, p=2., dim=-1)
        x = self.ffn(tokenized)
        z = self.conv(x.transpose(1, 2)).transpose(2, 1)  # Apply conv (B, L, D)
        z = self.feedforward(z)
        return z

class ConvolutionalEncoder(nn.Module):
    def __init__(self, vocab_size, dim, conv_kernel=[9], dilation=[1], symmetric=False, init_token: Optional[torch.Tensor]=None, normalized=True):
        super().__init__()
        self.dim = dim
        self.n_layers = len(dim)
        self.normalized = normalized
        init_dim = dim[0]

        if init_token is not None:
            init_token = self.preprocess_token(init_token, vocab_size)
            token_dim = init_token.shape[1]
            self.embedding = nn.Sequential(
                nn.Embedding.from_pretrained(init_token, freeze=False, padding_idx=tokens.index(pad_token)),
                nn.LayerNorm((token_dim))
            )
        else:
            token_dim = init_dim
            self.embedding = nn.Sequential(
                nn.Embedding(vocab_size, init_dim, padding_idx=tokens.index(pad_token)),
                nn.LayerNorm((token_dim))
            )

        self.ffn = nn.Sequential(
            nn.Linear(token_dim, init_dim*2),
            nn.GELU(),
            nn.Linear(init_dim*2, init_dim),
            nn.GELU(),
        )

        conv_class = SymmetricConv1d if symmetric else nn.Conv1d
        self.conv = self.make_conv_block(conv_class, conv_kernel[0], dim[0], dim[0], dilation[0], init_dim)
        for i in range(self.n_layers-1):
            in_dim = dim[i]
            out_dim = dim[i+1]
            conv = self.make_conv_block(conv_class, conv_kernel[i+1], in_dim, out_dim, dilation[i+1], init_dim)
            self.conv.extend(conv)

        ffn_dim = dim[-1]

        self.feedforward = nn.Sequential(
            nn.Linear(ffn_dim, ffn_dim * 2),
            nn.GELU(),
            nn.Linear(ffn_dim * 2, ffn_dim),
            nn.LayerNorm(ffn_dim),
            nn.GELU()
        )

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
        x = self.ffn(tokenized)
        B, K, L, Dtok = x.shape
        x = x.view(B * K, L, Dtok)
        #padding_mask = padding_mask.view(B * K, L)
        #x = self.pos_enc(x)
        z = self.conv(x.transpose(1, 2)).transpose(2, 1)  # Apply conv (B, L, D)
        z = self.feedforward(z)
        #z = z + x

        '''if self.n_layers > 0:
            z = self.pos_encoder(z)  # Add position info
            if padding_mask is not None:
                padding_mask = padding_mask == 0  # invert for PyTorch mask
            z = self.encoder(z, attn_mask=attention_mask, mask=padding_mask)'''
        _, _, Demb = z.shape
        z = z.view(B, K, L, Demb)
        #padding_mask = padding_mask.view(B, K, L)

        '''for layer in self.attn_layers:
            attn, norm, ff = layer
            x = attn(x) + x
            x = norm(x)
            x = ff(x) + x'''

        return z  # shape: (B, L, dim)

    def encode(self,input_ids):
        tokenized = self.embedding(input_ids.int())
        if self.normalized:
            tokenized = F.normalize(tokenized, p=2., dim=-1)
        x = self.ffn(tokenized)
        x = self.conv(x.transpose(1, 2)).transpose(2, 1)  # Apply conv (B, L, D)
        x = self.feedforward(x)
        return x

    def make_conv_block(self, conv_class, conv, indim, outdim, dilation, groups):
        padding = dilation * (conv - 1) // 2
        return nn.Sequential(
            conv_class(indim, outdim, kernel_size=conv, dilation=dilation, padding=padding, groups=groups),  # depthwise
            nn.GELU(),
            nn.BatchNorm1d(outdim)
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
    def __init__(self, vocab_size, dim, conv_kernel=[9], dilation=[1], symmetric=False, init_token: Optional[torch.Tensor]=None, normalized=True):
        super().__init__()
        self.dim = dim
        self.n_layers = len(dim)
        self.normalized = normalized
        if init_token is not None:
            init_token = self.preprocess_token(init_token, vocab_size)
            token_dim = init_token.shape[1]
            self.embedding = nn.Sequential(
                nn.Embedding.from_pretrained(init_token, freeze=False, padding_idx=tokens.index(pad_token)),
                nn.LayerNorm((max_len, token_dim))
            )
        else:
            token_dim = dim
            self.embedding = nn.Sequential(
                nn.Embedding(vocab_size, dim, padding_idx=tokens.index(pad_token)),
                nn.LayerNorm((max_len, token_dim))
            )
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, dim),
            nn.GELU(),
        )
        self.pos_enc = ConvolutionalPositionalEncoder(
            maxlen=max_length,
            dim=dim,
            conv_kernel=conv_kernel
        )

        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

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

    def encode(self,input_ids):
        tokenized = self.embedding(input_ids.int())
        if self.normalized:
            tokenized = F.normalize(tokenized, p=2., dim=-1)
        x = self.ffn(tokenized)
        z = self.conv(x.transpose(1, 2)).transpose(2, 1)  # Apply conv (B, L, D)
        z = self.feedforward(z)
        return z

class ProtConvColBERT(nn.Module):
    def __init__(
        self,
        dim=[32],
        conv_kernel=[5],
        dilation=[1],
        vocab_size=len(tokens),
        enc_type=0,
        symmetric=False,
        init_token: Optional[torch.Tensor]=None,
        outdim=512,
        normalized=True
    ):
        super().__init__()
        assert(len(dim) == len(conv_kernel) > 0 and len(dim) == len(conv_kernel) and len(dim) == len(dilation))
        if enc_type == 0:
            self.encoder = ConvolutionalEncoder(
                vocab_size=vocab_size,
                dim=dim,
                dilation=dilation,
                conv_kernel=conv_kernel,
                symmetric=symmetric,
                init_token=init_token,
                normalized=normalized
            )
            projection_dim = dim[-1]
        elif enc_type == 1:
            self.encoder = StackingConvolutionalEncoder(
                vocab_size=vocab_size,
                dim=dim,
                dilation=dilation,
                conv_kernel=conv_kernel,
                symmetric=symmetric,
                init_token=init_token,
                normalized=normalized
            )
            projection_dim = dim[-1]
        else:
            self.encoder = StackingEncoder(
                vocab_size=vocab_size,
                dim=dim,
                dilation=dilation,
                conv_kernel=conv_kernel,
                max_len=max_length,
                symmetric=symmetric,
                init_token=init_token,
                normalized=normalized
            )
            projection_dim = dim[-1]
        self.projection_head = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * projection_dim, outdim)
        )

    def forward(self, input_ids):
        x = self.encoder(input_ids)  # (B, 3, L, dim)
        x = self.projection_head(x)
        x = F.normalize(x, p=2., dim=-1, eps=1e-6)
        return x

    def encode(self, seq):
        x = self.encoder.encode(seq)
        return F.normalize(x, p=2., dim=-1, eps=1e-6)

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
            outdims=[512],
            normalized=True
        ):
        super().__init__()
        assert(len(dims) == len(dilations) and len(dims) == len(conv_kernels) and len(dims) == len(outdims))
        self.colbert = nn.ModuleList(
            ProtConvColBERT(
                dim=dim,
                vocab_size=vocab_size,
                conv_kernel=conv_kernel,
                dilation=dilation,
                enc_type=enc_type,
                symmetric=symmetric,
                init_token=init_token,
                normalized=normalized,
                outdim=outdim
            ) for dim, conv_kernel, dilation, outdim in zip(dims, conv_kernels, dilations, outdims)
        )
        self.num_enc = len(dims)
        self.simloss_scale = nn.Parameter(torch.tensor(5.0))
        self.guide_blosum = guide_blosum
        self.outdims = outdims
        self.normalized=normalized

    def forward(self, seq, seq_padmask, num_enc_used):
        mats = []
        for i in range(num_enc_used):
            m = self.colbert[i]
            seqemb = m(seq)
            mats.append(m.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,1,:,:], seq_padmask[:,0,:], seq_padmask[:,1,:]).unsqueeze(1))
        return torch.stack(mats,dim=1)

    @torch.jit.export
    def encode(self, seq):
        return [m.encode(seq) for m in self.colbert]
    
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
        return sum([self.sub_sim_loss(m.encoder.embedding[0].weight, target) for m in self.colbert])/len(self.outdims)

    def mmatch(self, sim, matchmask, nummatch, temp, abs_scale=0.1):
        scaled_sim = sim / temp
        mask = matchmask.sum(dim=2)
        pos_logits = (scaled_sim * matchmask).sum(dim=2)
        #assert((pos_logits <= scaled_sim.max(dim=2).values * mask).all())
        lse = torch.logsumexp(scaled_sim, dim=2)
        #assert((lse * mask >= scaled_sim.max(dim=2).values * mask).all())
        l = -(pos_logits - lse) * mask
        #assert((l >= 0).all())
        l = l.sum(dim=1)/nummatch
        aux_loss = torch.square((1-sim) * matchmask).sum(dim=(1,2))/nummatch
        return l + abs_scale * aux_loss

    def match_loss(self, mats, pair_match_mask, temp = 0.1, abs_scale=0.1):
        B, C, L, _ = mats.shape
        assert(mats.shape[3] == L and C == 1)
        mat_reshaped = mats.view((B*C,L,L))
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
        loss = torch.tensor(0.0, device=mats.device)
        refmask = pair_match_mask.float().clone()
        for i in range(num_enc_used):
            loss += self.match_loss(mats[:,i,:,:,:], refmask, temp=temp, abs_scale=abs_scale)
            refmask = (torch.abs(refmask - mats[:,i,:,:,:]) > (1-threshold)).float() * refmask
        return loss

    def loss_fn(self, mats, spanref, seqmatchmask, seqspanmask, pairmatchmask, temp=0.08, mask_padding=0, guide_scale=0.001, abs_scale=0.1, num_enc_used=1, threshold=0.4):
        mats_reduced = temp*torch.logsumexp(mats/temp,dim=1)
        coord_loss = self.match_loss_get(mats, pairmatchmask, temp=temp, abs_scale=abs_scale, num_enc_used=num_enc_used, threshold=threshold)
        q_idx = self.get_expected_idx(mats_reduced[:,0,:,:])
        d_idx = self.get_expected_idx(mats_reduced[:,0,:,:].transpose(1,2))
        q_order_loss = ProtConvColBERT.singular_locality_loss(q_idx, spanref[:,0,:], temp=0.08, padding=mask_padding)
        d_order_loss = ProtConvColBERT.singular_locality_loss(d_idx, spanref[:,1,:], temp=0.08, padding=mask_padding)
        order_loss = q_order_loss + d_order_loss
        sloss = self.sim_loss(device=mats.device)
        loss = coord_loss + guide_scale * sloss
        return loss, order_loss, coord_loss, self.signal_loss(mats_reduced, pairmatchmask), sloss
