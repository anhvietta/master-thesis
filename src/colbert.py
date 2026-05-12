import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder
from constants import max_length, attention_mask_window_size, blosum62_tensor, blosum62_std, blosum62_gttl
from utils import tokens, pad_token
import math
import random
from typing import Optional

class ConvBERTBlock(nn.Module):
    def __init__(self, dim, n_heads, kernel_size=9, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

        # Depthwise separable convolution
        self.conv_proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

        # Mixing attention and convolution features
        self.mix = nn.Linear(2 * dim, dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask: Optional[torch.Tensor]=None):
        # x: (B, L, dim)
        res = x

        # Multi-head attention
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)  # (B, L, dim)

        # Depthwise conv over sequence
        x_conv = x.transpose(1, 2)  # (B, dim, L)
        conv_out = self.conv_proj(x_conv).transpose(1, 2)  # (B, L, dim)

        # Concatenate and mix
        mixed = torch.cat([attn_out, conv_out], dim=-1)  # (B, L, 2*dim)
        x = self.mix(mixed)  # (B, L, dim)

        x = self.norm1(x + res)

        # Feed-forward + residual
        res2 = x
        x = self.ffn(x)
        x = self.norm2(x + res2)

        return x

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
    def __init__(self, vocab_size, dim, conv_kernel=9, dropout=0.1, n_layers=2, n_heads=8, max_len=max_length, symmetric=True, init_token: Optional[torch.Tensor]=None):
        super().__init__()
        assert conv_kernel % 2 == 1, "Kernel size should be odd for symmetric windowing."
        self.symmetric=symmetric
        #self.embedding = nn.Embedding.from_pretrained(torch.tensor blosum62_gttl, freeze=False, padding_idx=tokens.index(pad_token))
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

    def forward(self, input_ids, padding_mask, attention_mask: Optional[torch.Tensor]=None):
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

class TinyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, dim, conv_kernel=9, dropout=0.1, n_layers=2, n_heads=8, max_len=max_length, symmetric=False, init_token: Optional[torch.Tensor]=None, normalized=True):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
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
        '''self.pos_enc = RelativePositionalEncoding(
            maxlen = max_length,
            d_model = dim,
            max_relative_distance=conv_kernel
        )'''

        conv_class = SymmetricConv1d if symmetric else nn.Conv1d

        # Convolutional block (depthwise)
        self.conv = nn.Sequential(
            conv_class(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim),  # depthwise
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )
        for i in range(n_layers-1):
            in_dim = int(dim*(2**i))
            out_dim = int(dim*(2**(i+1)))

            conv = nn.Sequential(
                conv_class(in_dim, out_dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=in_dim),  # depthwise
                nn.GELU(),
                nn.BatchNorm1d(out_dim),
            )
            self.conv.extend(conv)
        '''self.conv = nn.Sequential(
            conv_class(dim, dim * 2, kernel_size=1),         # expand
            nn.GELU(),
            nn.BatchNorm1d(dim * 2),
            conv_class(dim * 2, dim * 2, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim * 2),  # depthwise
            nn.GELU(),
            nn.BatchNorm1d(dim * 2),
            conv_class(dim * 2, dim, kernel_size=1),         # project back
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )'''

        ffn_dim = dim*2**(n_layers-1)

        self.feedforward = nn.Sequential(
            nn.Linear(ffn_dim, ffn_dim * 2),
            nn.GELU(),
            nn.Linear(ffn_dim * 2, ffn_dim),
            nn.LayerNorm(ffn_dim)
        )
        
        '''if n_layers > 0:
            # Positional encoding
            self.pos_encoder = PositionalEncoding(dim, max_len=max_len)
            self.encoder=Encoder(
                dim=dim,
                depth=n_layers,
                heads=n_heads,
                ff_mult=4,
                layer_dropout=dropout,
                ff_dropout=dropout,
                attn_dropout=dropout,
                attn_flash = True,
                use_simple_rmsnorm = True,
                ff_glu = True,
                ff_no_bias = True,
                alibi_pos_bias = True, # turns on ALiBi positional embedding
            )'''

    def preprocess_token(self, init_token, vocab_size):
        init_token = torch.tensor(init_token, dtype=torch.float)
        if init_token.shape[0] < vocab_size:
            init_token = F.pad(init_token, (0,0,0,vocab_size-init_token.shape[0]), 'constant', 0)
        elif init_token.shape[0] > vocab_size:
            raise RuntimeError("Incompatible init tokens")
        assert(init_token.shape[0] == vocab_size)
        return init_token

    def forward(self, input_ids, padding_mask, attention_mask: Optional[torch.Tensor]=None):
        tokenized = self.embedding(input_ids.int())
        if self.normalized:
            tokenized = F.normalize(tokenized, p=2., dim=-1)
        x = self.ffn(tokenized)
        B, K, L, Dtok = x.shape
        x = x.view(B * K, L, Dtok)
        padding_mask = padding_mask.view(B * K, L)
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
        padding_mask = padding_mask.view(B, K, L)

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
    def __init__(self, vocab_size, dim, conv_kernel=9, dropout=0.1, n_layers=2, n_heads=8, max_len=max_length, symmetric=False, init_token: Optional[torch.Tensor]=None, normalized=True):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
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

        '''conv_class = SymmetricConv1d if symmetric else nn.Conv1d

        # Convolutional block (depthwise)
        self.conv = nn.Sequential(
            conv_class(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim),  # depthwise
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )'''
        '''self.conv = nn.Sequential(
            conv_class(dim, dim * 2, kernel_size=1),         # expand
            nn.GELU(),
            nn.BatchNorm1d(dim * 2),
            conv_class(dim * 2, dim * 2, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim * 2),  # depthwise
            nn.GELU(),
            nn.BatchNorm1d(dim * 2),
            conv_class(dim * 2, dim, kernel_size=1),         # project back
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )'''

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

    def forward(self, input_ids, padding_mask, attention_mask: Optional[torch.Tensor]=None):
        tokenized = self.embedding(input_ids.int())
        if self.normalized:
            tokenized = F.normalize(tokenized, p=2., dim=-1)
        x = self.ffn(tokenized)
        B, K, L, Dtok = x.shape
        x = x.view(B * K, L, Dtok)
        padding_mask = padding_mask.view(B * K, L)
        z = self.pos_enc(x)
        z = self.feedforward(z)
        #z = z + x
        _, _, Demb = z.shape
        z = z.view(B, K, L, Demb)
        padding_mask = padding_mask.view(B, K, L)
        return z  # shape: (B, L, dim)

    def encode(self,input_ids):
        tokenized = self.embedding(input_ids.int())
        if self.normalized:
            tokenized = F.normalize(tokenized, p=2., dim=-1)
        x = self.ffn(tokenized)
        z = self.conv(x.transpose(1, 2)).transpose(2, 1)  # Apply conv (B, L, D)
        z = self.feedforward(z)
        return z

def mask_sigmoid(x, a=10, temp=0.05):
    return a * torch.sigmoid(x / temp)

class BlosumScorer(nn.Module):
    def __init__(self, blosum_matrix, conv_kernel):
        super().__init__()
        self.matrix = nn.Parameter(torch.tensor(blosum_matrix, dtype=torch.float32))
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

class ProtConvColBERT(nn.Module):
    def __init__(self, dim, vocab_size=len(tokens), conv_kernel=17, dropout=0.1, n_layers=2, n_heads=8, max_len=max_length, locality_coef=10, enc_type=0, symmetric=False, init_token: Optional[torch.Tensor]=None, normalized=True):
        super().__init__()
        if enc_type == 0:
            self.encoder = TinyTransformerEncoder(
                vocab_size=vocab_size,
                dim=dim,
                conv_kernel=conv_kernel,
                dropout=dropout,
                n_layers=n_layers,
                n_heads=n_heads,
                max_len=max_length,
                symmetric=symmetric,
                init_token=init_token,
                normalized=normalized
            )
            project_dim = dim
        elif enc_type == 1:
            self.encoder = StackingConvolutionalEncoder(
                vocab_size=vocab_size,
                dim=dim,
                conv_kernel=conv_kernel,
                dropout=dropout,
                n_layers=n_layers,
                n_heads=n_heads,
                max_len=max_length,
                symmetric=symmetric,
                init_token=init_token,
                normalized=normalized
            )
            project_dim = dim
        else:
            self.encoder = StackingEncoder(
                vocab_size=vocab_size,
                dim=dim,
                conv_kernel=conv_kernel,
                dropout=dropout,
                n_layers=n_layers,
                n_heads=n_heads,
                max_len=max_length,
                symmetric=symmetric,
                init_token=init_token,
                normalized=normalized
            )
            project_dim = int(dim * conv_kernel)
        #self.project = nn.Linear(project_dim, project_dim)  # can also use identity if dim fixed
        #self.locality_coef = locality_coef
        #self.del_per_aa = nn.Parameter(torch.zeros(vocab_size))
        self.blosum_scorer = BlosumScorer(blosum62_gttl, conv_kernel)
        #self.blosum = nn.Parameter(torch.tensor(blosum62_gttl, dtype=torch.float32))

    '''def create_blosum_mask(self, input_ids):
        assert(input_ids.size(1) == 3)
        input_ids = input_ids.int()
        q = input_ids[:,0,:]
        p = input_ids[:,1,:]
        n = input_ids[:,2,:]

        B, L = q.shape

        def scores(s1, s2):
            #rows = self.blosum[s1.reshape(-1)].reshape(s1.size(0), s1.size(1), -1)
            rows = torch.index_select(self.blosum, 0, s1.reshape(-1)).reshape(B, L, -1)
            s2_exp = s2.unsqueeze(1).expand(-1, s1.size(1), -1)
            scores = torch.gather(rows, 2, s2_exp)
            return scores

        qp = scores(q,p)
        qn = scores(q,n)

        return torch.stack((qp,qn), dim=1)'''

    def forward(self, input_ids, attention_mask, padding_mask):
        x = self.encoder(input_ids, padding_mask, attention_mask)  # (B, 3, L, dim)
        #print(x.shape)
        #x = self.project(x)
        x = F.normalize(x, p=2., dim=-1, eps=1e-6)
        #x = (x - x.min())/(x.max() - x.min())
        #del_logits = self.del_per_aa[input_ids.int()]
        #blosum_mask = self.create_blosum_mask(input_ids)
        blosum_mask = self.blosum_scorer(input_ids)
        return x, blosum_mask

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

class CoordRegressor(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=8, conv_kernel=5):
        super(CoordRegressor, self).__init__()
        '''self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=conv_kernel, stride=2, padding=conv_kernel // 2),  # (256x256)
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=conv_kernel, stride=2, padding=conv_kernel // 2),  # (128x128)
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=conv_kernel, stride=2, padding=conv_kernel // 2),  # (64x64)
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: (128, 1, 1)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),              # (128,)
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, 4),          # Output: [x1, y1, x2, y2]
            nn.Sigmoid()               # Normalize to [0, 1]
        )'''

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=conv_kernel, padding=conv_kernel//2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )


    def forward(self, x):
        start = self.decoder(self.encoder(x))
        end = self.decoder(self.encoder(torch.flip(x,[2,3])))
        end = torch.flip(end,[2,3])
        return torch.cat((start,end), dim=1)

    @staticmethod
    def loss_fn(heatmaps, gaussian_maps):
        B, C, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, C, -1)
        '''softmax = F.softmax(heatmaps, dim=-1)
        softmax = softmax.view(B, C, H, W)
        '''
        log_probs = F.log_softmax(heatmaps, dim=-1).view(B*C,H,W)
        gaussian_maps = gaussian_maps.view(B*C,H,W)
        return F.kl_div(log_probs,gaussian_maps,reduction='batchmean')
        #return F.mse_loss(softmax,gaussian_maps,reduction='mean')
        #return F.cross_entropy(feature_map,gaussian_maps,reduction='mean')

    @staticmethod
    def map_to_coords(heatmaps):
        B, C, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, C, -1)
        softmax = F.softmax(heatmaps, dim=-1)
        softmax = softmax.view(B, C, H, W)

        # Coordinate grids
        xs = torch.linspace(0, 1, W, device=heatmaps.device)
        ys = torch.linspace(0, 1, H, device=heatmaps.device)
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')

        xs = xs[None, None, :, :].expand_as(softmax)  # (B, C, H, W)
        ys = ys[None, None, :, :].expand_as(softmax)

        x = torch.sum(xs * softmax, dim=(2, 3))
        y = torch.sum(ys * softmax, dim=(2, 3))

        coords = torch.stack([x, y], dim=2)  # (B, C, 2)

        return coords

class AlignmentRegressor(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=8, conv_kernel=7, max_len=max_length):
        super(AlignmentRegressor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=conv_kernel, padding=conv_kernel//2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=conv_kernel, padding=conv_kernel//2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        q_aln = self.cnn(x)
        #weights = torch.softmax(q_aln, dim=3)
        #q_aln = torch.sum(q_aln * weights, dim=3)
        q_aln, _ = torch.max(q_aln, dim=3)
        d_aln = self.cnn(x.transpose(2,3))
        #weights = torch.softmax(d_aln, dim=3)
        #d_aln = torch.sum(d_aln * weights, dim=3)
        d_aln, _ = torch.max(d_aln, dim=3)
        return torch.cat((q_aln,d_aln),dim=1)

    @staticmethod
    def loss_fn(aln, alnref):
        def soft_high_count(x, threshold=0.85, sharpness=20):
            return torch.sigmoid(sharpness * (x - threshold)).sum()
        def count_loss(x,y):
            return (soft_high_count(x) - soft_high_count(y)).pow(2)
        aln_loss = F.mse_loss(aln, alnref)
        #match_loss = F.mse_loss(aln[:,0,:].sum(dim=-1), aln[:,1,:].sum(dim=-1)) + F.mse_loss(aln[:,2,:].sum(dim=-1), aln[:,3,:].sum(dim=-1))
        #match_loss = count_loss(aln[:,0,:], aln[:,1,:]) + count_loss(aln[:,2,:], aln[:,3,:])
        match_loss = torch.tensor(0)
        return aln_loss + match_loss, aln_loss, match_loss

class LocAligner(nn.Module):
    def __init__(self,
            dim,
            vocab_size=len(tokens),
            conv_kernel=17,
            dropout=0.1,
            n_layers=2,
            n_heads=8,
            coord_channels=8,
            coord_kernel=7,
            aln_channels=8,
            aln_kernel=7,
            max_len=max_length
        ):
        super().__init__()
        self.colbert = ProtConvColBERT(
            dim=dim,
            vocab_size=vocab_size,
            conv_kernel=conv_kernel,
            dropout=dropout,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len
        )

        self.coord_regressor = CoordRegressor(
            hidden_channels=coord_channels,
            conv_kernel=coord_kernel
        )

        self.alignment_regressor = AlignmentRegressor(
            hidden_channels=aln_channels,
            conv_kernel=aln_kernel
        )

    def make_differentiable_mask(self, heatmaps, temp):
        H, W = max_length, max_length
        coords = CoordRegressor.map_to_coords(heatmaps)
        B = coords.size(0)
        start_xy = coords[:, 0, :]
        end_xy = coords[:, 1, :]

        # Convert normalized [0,1] coords to pixel indices
        x0 = coords[:, 0, 0] * (W - 1)
        y0 = coords[:, 0, 1] * (H - 1)
        x1 = coords[:, 1, 0] * (W - 1)
        y1 = coords[:, 1, 1] * (H - 1)

        # Create pixel grids (H, W)
        ys = torch.linspace(0, H - 1, H, device=start_xy.device).view(1, H, 1)
        xs = torch.linspace(0, W - 1, W, device=start_xy.device).view(1, 1, W)

        # Expand to (B, H, W)
        xs = xs.expand(B, -1, -1)
        ys = ys.expand(B, -1, -1)

        # Expand coords to (B, H, W)
        x0 = x0.view(B, 1, 1)
        x1 = x1.view(B, 1, 1)
        y0 = y0.view(B, 1, 1)
        y1 = y1.view(B, 1, 1)

        # Compute sigmoid-based edges
        mask_x = torch.sigmoid((xs - x0)/temp) * torch.sigmoid((x1 - xs)/temp)
        mask_y = torch.sigmoid((ys - y0)/temp) * torch.sigmoid((y1 - ys)/temp)
        mask = mask_x * mask_y  # (B, H, W)
        return mask


    def forward(self, seq, seq_attnmat, seq_padmask, temp=0.2):
        seqemb, seqdel = self.colbert(seq, seq_attnmat, seq_padmask)
        qp_mat = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,1,:,:], seq_padmask[:,0,:], seq_padmask[:,1,:], unsqueeze=1)
        qn_mat = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,2,:,:], seq_padmask[:,0,:], seq_padmask[:,2,:], unsqueeze=1)
        pgmap = self.coord_regressor(qp_mat)
        ngmap = self.coord_regressor(qn_mat)
        qp_mask = self.make_differentiable_mask(pgmap, temp)
        qn_mask = self.make_differentiable_mask(ngmap, temp)
        qp_mat_masked = qp_mat * qp_mask.unsqueeze(1)
        qn_mat_masked = qn_mat * qn_mask.unsqueeze(1)
        paln_logits = self.alignment_regressor(qp_mat_masked)
        naln_logits = self.alignment_regressor(qn_mat_masked)
        qpaln_prob = F.softmax(torch.stack([paln_logits[:,0,:],seqdel[:,0,:]],dim=2), dim=2)[:,:,0]
        paln_prob = F.softmax(torch.stack([paln_logits[:,1,:],seqdel[:,1,:]],dim=2), dim=2)[:,:,0]
        qnaln_prob = F.softmax(torch.stack([naln_logits[:,0,:],seqdel[:,0,:]],dim=2), dim=2)[:,:,0]
        naln_prob = F.softmax(torch.stack([naln_logits[:,1,:],seqdel[:,2,:]],dim=2), dim=2)[:,:,0]
        aln_prob = torch.stack((qpaln_prob,paln_prob,qnaln_prob,naln_prob), dim=1)
        return torch.cat((qp_mat,qn_mat), dim=1), torch.cat((pgmap,ngmap),dim=1), aln_prob

    @staticmethod
    def score_aln(qaln, daln):
        n = qaln.size(1) + daln.size(1) - 1
        A = torch.fft.rfft(qaln,n)
        B = torch.fft.rfft(daln,n)
        corr = torch.fft.irfft(A * torch.conj(B), n)
        max_corr, _ = corr.max(dim=1)
        return max_corr
        #return qaln.mean(dim=1) * daln.mean(dim=1)

    @staticmethod
    def loss_fn(hmap, hmapref, aln, alnref, margin=0.05):
        coord_loss = CoordRegressor.loss_fn(hmap, hmapref)
        total_aln_loss, refaln_loss, match_loss = AlignmentRegressor.loss_fn(aln, alnref)
        pos_score = LocAligner.score_aln(aln[:,0,:], aln[:,1,:])
        neg_score = LocAligner.score_aln(aln[:,2,:], aln[:,3,:])
        target = torch.ones_like(pos_score)
        #ranking_loss = F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)
        ranking_loss = F.binary_cross_entropy_with_logits((pos_score - neg_score) - margin, target)
        return coord_loss, total_aln_loss, ranking_loss, refaln_loss, match_loss, pos_score, neg_score


# --- Building blocks ---
class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class RollUpHead(nn.Module):
    def __init__(self, c_in, hid=16):
        super().__init__()
        self.col = nn.Sequential(
            nn.Conv1d(c_in, hid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hid, 1, 1)
        )
    def forward(self, feat):  # (N,C,H,W)
        N, C, H, W = feat.shape
        feat = feat.view((N,C*H,W))
        col_feat = feat.mean(dim=-1).view(N,C,H)  # (N,C,H)
        x_logits = self.col(col_feat).squeeze(1)  # (N,H)
        return x_logits

class AlignmentAxisUNet(nn.Module):
    def __init__(self, in_ch=1, base=4, depth=4, L=max_length):
        """
        Args:
            in_ch: number of input channels
            base: number of channels in the first layer
            depth: number of encoder/decoder levels
            L: output length (image width/height)
        """
        super().__init__()
        self.depth = depth
        self.L = L

        # --- Encoder ---
        c_in = base
        self.enc_blocks = DoubleConv(in_ch, c_in)
        self.down_blocks = nn.ModuleList()
        for _ in range(depth):
            self.down_blocks.append(Down(c_in, c_in * 2))
            c_in *= 2

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up_blocks.append(Up(c_in, c_in // 2))
            c_in //= 2

        # --- Heads at each scale ---
        self.heads = nn.ModuleList()
        self.heads.append(RollUpHead(base * (2**depth)))  # bottleneck head
        for i in reversed(range(depth)):
            self.heads.append(RollUpHead(base * (2**i)))

        '''self.ffn = nn.Sequential(
            nn.Linear(depth+1, 1),
            nn.ReLU(inplace=True)
        )'''

    def _resize_1d(self, logits, L_out):
        return F.interpolate(logits.unsqueeze(1), size=L_out, mode='linear',
                             align_corners=False)

    def forward(self, x):
        skips = []
        # Encoder
        out = self.enc_blocks(x)
        skips.append(out)
        for i in range(self.depth):
            out = self.down_blocks[i](out)
            skips.append(out)

        # Decoder
        d = skips[-1]
        dec_feats = []
        for i in range(self.depth):
            skip = skips[self.depth - 1 - i]
            d = self.up_blocks[i](d, skip)
            dec_feats.append(d)

        # Roll-up at multiple scales
        feats = [skips[-1]] + dec_feats
        xs = [self._resize_1d(self.heads[i](f), self.L) for i, f in enumerate(feats)]
        #x_logits = torch.cat(xs, dim=1).transpose(1,2)  # (N,L,depth+1)
        #x_logits = self.ffn(x_logits).squeeze(2) #(N,L)
        x_logits = torch.stack(xs, dim=0).sum(dim=0)  # (N,L)
        return x_logits

class AlignmentUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=2, depth=3, bilinear=False):
        super().__init__()
        self.depth = depth
        self.bilinear = bilinear
        # Encoder
        self.inc = DoubleConv(in_channels, base_filters)
        self.downs = nn.ModuleList()
        filters = base_filters
        for i in range(depth):
            self.downs.append(Down(filters, filters * 2))
            filters *= 2

        # Bottleneck is just the last output of downs
        self.bottleneck_channels = filters

        # Decoder
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(filters, filters // 2, bilinear))
            filters //= 2

        self.outc = OutConv(filters, out_channels)

    def forward(self, x):
        B = x.size(0)
        enc_feats = []
        x = self.inc(x)
        enc_feats.append(x)

        # Encoder path
        for down in self.downs:
            x = down(x)
            enc_feats.append(x)

        # Decoder path (skip connections from encoder)
        for i, up in enumerate(self.ups):
            x = up(x, enc_feats[-(i+2)])  # skip connection

        logits = self.outc(x)
        return logits

def flip_simmat(S, len_x, len_y):
    mat = S.squeeze(1)
    B, Lx, Ly = mat.shape
    device = mat.device

    # indices along x and y
    idx_x = torch.arange(Lx, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)  # (B, Lx)
    idx_y = torch.arange(Ly, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)  # (B, Ly)

    # reversed indices per batch
    rev_idx_x = len_x.unsqueeze(1) - 1 - idx_x
    rev_idx_y = len_y.unsqueeze(1) - 1 - idx_y

    # mask out padding positions (leave them unchanged)
    rev_idx_x = torch.where(idx_x < len_x.unsqueeze(1), rev_idx_x, idx_x)
    rev_idx_y = torch.where(idx_y < len_y.unsqueeze(1), rev_idx_y, idx_y)

    # clip to valid range
    rev_idx_x = torch.clamp(rev_idx_x, min=0)
    rev_idx_y = torch.clamp(rev_idx_y, min=0)

    # gather rows then columns
    mat_flipped = torch.gather(mat, 1, rev_idx_x.unsqueeze(2).expand(-1, -1, Ly))
    mat_flipped = torch.gather(mat_flipped, 2, rev_idx_y.unsqueeze(1).expand(-1, Lx, -1))

    return mat_flipped.unsqueeze(1)

class CoordRegressorUNet(nn.Module):
    def __init__(self, in_ch=1, base=4, depth=4, L=max_length):
        """
        Args:
            in_ch: number of input channels
            base: number of channels in the first layer
            depth: number of encoder/decoder levels
            L: output length (image width/height)
        """
        super().__init__()
        self.depth = depth
        self.L = L

        # --- Encoder ---
        c_in = base
        self.enc_blocks = DoubleConv(in_ch, c_in)
        self.down_blocks = nn.ModuleList()
        for _ in range(depth):
            self.down_blocks.append(Down(c_in, c_in * 2))
            c_in *= 2

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up_blocks.append(Up(c_in, c_in // 2))
            c_in //= 2

        # --- Heads at each scale ---
        self.heads = nn.ModuleList()
        self.heads.append(RollUpHead(base * (2**depth)))  # bottleneck head
        for i in reversed(range(depth)):
            self.heads.append(RollUpHead(base * (2**i)))

    def _resize_1d(self, logits, L_out):
        return F.interpolate(logits.unsqueeze(1), size=L_out, mode='linear',
                             align_corners=False)

    def regress(self, x):
        skips = []
        # Encoder
        out = self.enc_blocks(x)
        skips.append(out)
        for i in range(self.depth):
            out = self.down_blocks[i](out)
            skips.append(out)

        # Decoder
        d = skips[-1]
        dec_feats = []
        for i in range(self.depth):
            skip = skips[self.depth - 1 - i]
            d = self.up_blocks[i](d, skip)
            dec_feats.append(d)

        # Roll-up at multiple scales
        feats = [skips[-1]] + dec_feats
        xs = [self._resize_1d(self.heads[i](f), self.L) for i, f in enumerate(feats)]
        #x_logits = torch.cat(xs, dim=1).transpose(1,2)  # (N,L,depth+1)
        #x_logits = self.ffn(x_logits).squeeze(2) #(N,L)
        x_logits = torch.stack(xs, dim=0).sum(dim=0)  # (N,L)
        return x_logits

    def forward(self, x, qlen, dlen):
        B = x.size(0)
        flipped_x = flip_simmat(x, dlen, qlen)
        x_start = self.regress(x)
        y_start = self.regress(x.transpose(2,3))
        x_end = self.regress(flipped_x)
        y_end = self.regress(flipped_x.transpose(2,3))
        logits = torch.cat((x_start, y_start, x_end, y_end),dim=1)
        sigmoid = F.softmax(logits,dim=2)
        idx = torch.arange(max_length, dtype=sigmoid.dtype, device=sigmoid.device).expand(B,4,-1)
        positions = (sigmoid * idx).sum(dim=2)
        return positions #(B,4)

    @staticmethod
    def loss_fn(positions, reference):
        ref_loss = F.mse_loss(positions, reference)
        return ref_loss

# ---------- clDice loss ----------
# Ref: "clDice - a novel connectivity-preserving loss function"
# https://arxiv.org/abs/2003.07311
def soft_skeletonize(x, iters= 50):
    """
    Differentiable approximation of skeletonization.
    x: tensor of shape (N, 1, H, W), values in [0,1]
    """
    p = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    s = F.relu(p - x)
    for _ in range(iters):
        p = F.max_pool2d(s, kernel_size=3, stride=1, padding=1)
        s = F.relu(p - s)
    return s

def cldice_loss(pred, target, iters = 10, eps = 1e-6):
    """
    pred: prediction in [0,1], shape (N,1,H,W)
    target: ground truth mask {0,1}, shape (N,1,H,W)
    """
    skel_pred = soft_skeletonize(pred, iters)
    skel_gt   = soft_skeletonize(target, iters)

    # precision: fraction of predicted skeleton covered by GT
    tprec = (torch.sum(skel_pred * target) + eps) / (torch.sum(skel_pred) + eps)
    # sensitivity: fraction of GT skeleton covered by prediction
    tsens = (torch.sum(skel_gt * pred) + eps) / (torch.sum(skel_gt) + eps)

    cl_dice = 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + eps)
    return cl_dice

# ---------- Soft Hausdorff loss ----------
# Differentiable approximation based on Chamfer distance
# (instead of strict max, use min + averaging)
def hausdorff_loss(pred, target, eps = 1e-6):
    """
    pred: probability map (N,1,H,W)
    target: binary mask (N,1,H,W)
    """
    pred_points = pred.view(pred.size(0), -1)
    target_points = target.view(target.size(0), -1)

    # Normalize to probability distributions
    pred_points = pred_points / (pred_points.sum(dim=1, keepdim=True) + eps)
    target_points = target_points / (target_points.sum(dim=1, keepdim=True) + eps)

    # Coordinates grid
    N, _, H, W = pred.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=pred.device),
        torch.arange(W, device=pred.device),
        indexing="ij"
    )
    coords = torch.stack([yy, xx], dim=-1).float()  # (H, W, 2)
    coords = coords.view(-1, 2)  # (HW, 2)

    # Distance matrix (HW x HW)
    dists = torch.cdist(coords[None, ...], coords[None, ...])  # (1,HW,HW)

    # Chamfer-like loss
    forward = torch.sum(pred_points[:, :, None] * torch.min(dists, dim=2).values, dim=1)
    backward = torch.sum(target_points[:, :, None] * torch.min(dists, dim=2).values, dim=1)

    return (forward + backward).mean()

class AlignmentAE(nn.Module):
    @staticmethod
    def get_enc_conv(in_channels, out_channels, kernel, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_channels,eps=10**-3),
            nn.GELU()
        )

    @staticmethod
    def get_dec_conv(in_channels, out_channels, kernel, padding, index):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                padding=padding,
                stride=1,
                output_padding=0,
                dilation=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    @staticmethod
    def get_enc_pool(kernel, padding, stride=2):
        return nn.MaxPool2d(kernel_size=kernel, padding=padding, stride=stride, return_indices=True)

    @staticmethod
    def get_dec_pool(kernel, padding, stride=2):
        return nn.MaxUnpool2d(kernel_size=kernel, padding=padding, stride=stride)

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_filters=2,
        enconv_kernel=3,
        deconv_kernel=3,
        pool_kernel=2,
        pool_padding=0,
        depth=5,
        dec_dropout=0.1
    ):
        super(AlignmentAE, self).__init__()
        self.num_blocks = depth
        self.enc_conv = nn.ModuleList([
            AlignmentAE.get_enc_conv(
                in_channels=in_channels,
                out_channels=base_filters,
                kernel=enconv_kernel,
                padding=enconv_kernel//2
            )] + [
            AlignmentAE.get_enc_conv(
                in_channels=base_filters*2**i,
                out_channels=base_filters*2**(i+1),
                kernel=enconv_kernel,
                padding=enconv_kernel//2
            ) for i in range(depth)
        ])
        self.enc_pool = AlignmentAE.get_enc_pool(
            kernel=pool_kernel,
            padding=pool_padding
        )
        self.dec_conv = nn.ModuleList([
            AlignmentAE.get_dec_conv(
                in_channels=base_filters*2**i,
                out_channels=base_filters*2**(i-1),
                kernel=deconv_kernel,
                padding=deconv_kernel//2,
                index=i
            ) for i in range(depth,0,-1)
        ] + [
            AlignmentAE.get_dec_conv(
                in_channels=base_filters,
                out_channels=out_channels,
                kernel=deconv_kernel,
                padding=deconv_kernel//2,
                index=depth+1
        )])
        self.dec_pool = AlignmentAE.get_dec_pool(
            kernel=pool_kernel,
            padding=pool_padding
        )
        self.dec_dropout = nn.Dropout(dec_dropout)

    def encode(self, z):
        pool_idx = []
        for i in range(self.num_blocks+1):
            z = self.enc_conv[i](z)
            z, idx = self.enc_pool(z)
            pool_idx.append(idx)
        return z, pool_idx

    def decode(self, recon, pool_idx):
        for i in range(self.num_blocks+1):
            recon = self.dec_pool(recon, pool_idx[i])
            recon = self.dec_conv[i](recon)
            recon = self.dec_dropout(recon)
        return recon

    def forward(self, x):
        z, pool_idx = self.encode(x)
        pool_idx.reverse()
        out = self.decode(z, pool_idx)
        return out

class SeqMatcher(nn.Module):
    def __init__(self,
            dim,
            vocab_size=len(tokens),
            conv_kernel=17,
            dropout=0.1,
            n_layers=2,
            n_heads=8,
            aln_channels=2,
            aln_layers=3,
            max_len=max_length):
        super().__init__()
        self.colbert = ProtConvColBERT(
            dim=dim,
            vocab_size=vocab_size,
            conv_kernel=conv_kernel,
            dropout=dropout,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len
        )

        self.aln_cnn = AlignmentAxisUNet(
            base=aln_channels,
            depth=aln_layers
        )

        self.gap_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, seq, seq_attnmat, seq_padmask, transpose=False, temp=0.2):
        B = seq.size(0)
        seqemb, seqdel = self.colbert(seq, seq_attnmat, seq_padmask)
        qp_mat = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,1,:,:], seq_padmask[:,0,:], seq_padmask[:,1,:]).unsqueeze(1)
        qn_mat = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,2,:,:], seq_padmask[:,0,:], seq_padmask[:,2,:]).unsqueeze(1)
        if transpose:
            qp_mat, qn_mat = qp_mat.transpose(2,3), qn_mat.transpose(2,3)
        mats = torch.cat((qp_mat, qp_mat.transpose(2,3), qn_mat, qn_mat.transpose(2,3)),dim = 1).view((B * 4, 1, max_length, max_length))
        alns = self.aln_cnn(mats).view((B,4,max_length))
        return alns, qp_mat, qn_mat
        #naln_map = self.aln_cnn(qn_mat)
        #pgap_score = SeqMatcher.score_gap(paln_map)
        #ngap_score = SeqMatcher.score_gap(naln_map)
        #psim_score = (qp_mat * F.sigmoid(paln_map)).reshape((B,-1)).sum(dim=-1)
        #nsim_score = (qn_mat * F.sigmoid(naln_map)).reshape((B,-1)).sum(dim=-1)
        #pos_score = psim_score - self.gap_scale * pgap_score
        #neg_score = nsim_score - self.gap_scale * ngap_score
        #return paln_map, naln_map, pos_score, neg_score

    @staticmethod
    def score_gap(aln_map):
        def create_soft_mask(seq_aln, threshold=0.5, sharpness=20.):
            soft = F.sigmoid(sharpness * (seq_aln - threshold))  # soft threshold
            # cumulative product from left and right ensures one contiguous block
            left = soft.cummax(dim=1).values
            right = soft.flip(1).cummax(dim=1).values.flip(1)
            return (left * right)

        def score_seq_gap(amap, gapopen_f=10, gap_f=1):
            seq_aln = amap.squeeze(1).sum(dim=-1)
            soft_mask = create_soft_mask(seq_aln)
            gaps = F.relu((1-seq_aln) * soft_mask)

            # difference detects "rising edges"
            diff = gaps[:,1:] - gaps[:,:-1]
            # count edges with ReLU to ignore negative slopes
            block_starts = F.relu(diff).sum(dim=-1)

            gap_score = gap_f * gaps.sum(dim=-1) + gapopen_f * block_starts
            return gap_score

        return score_seq_gap(aln_map) + score_seq_gap(aln_map.transpose(2,3))

    @staticmethod
    def bce_dice(preds, targets, smooth=1e-6, alpha_F=0.8, gamma_F=2, alpha_T=1., beta_T=1.):
        #B, C, L = targets.shape
        #preds = preds.view((B*C,L))
        #targets = targets.to(dtype=preds.dtype,device=preds.device).reshape((B*C,L))

        # BCE
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets)
        '''BCE = F.binary_cross_entropy_with_logits(preds, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_bce_loss = alpha_F * (1-BCE_EXP)**gamma_F * BCE'''
        # Dice
        preds = F.sigmoid(preds)
        '''intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (
            preds.sum() + targets.sum() + smooth
        )'''
        TP = (preds * targets).sum()
        FP = ((1-targets) * preds).sum()
        FN = (targets * (1-preds)).sum()
        aux = alpha_T*FP + beta_T*FN + smooth

        tversky_dice_loss = aux / (TP + aux + smooth)
        #preds = preds.view((B,C,L)).unsqueeze(1)
        #targets = targets.view((B,C,L)).unsqueeze(1)

        #cldice = cldice_loss(preds, targets)
        #hausdorff = hausdorff_loss(preds, targets)

        return tversky_dice_loss + bce_loss, bce_loss, tversky_dice_loss

    @staticmethod
    def kl_loss(log_preds, targets):
        # KL divergence (target * log(target / pred))
        # Using torch.kl_div expects inputs=log_probs, targets=prob
        loss = F.kl_div(log_preds, targets, reduction='batchmean')
        return loss

    @staticmethod
    def auxiliary_cov_entropy_loss(pred_probs, hard_targets, eps=1e-6):
        inter = (pred_probs * hard_targets).sum(dim=1)
        denom = hard_targets.sum(dim=1).clamp(min=eps)
        cov = 1.0 - (inter / denom).mean()
        entropy = -(pred_probs * (pred_probs + eps).log()).sum(dim=1).mean()
        return cov + 0.001 * entropy

    @staticmethod
    def soft_heatmap_loss(preds, targets, hard_targets):
        preds = preds.squeeze(1)
        B, H, W = preds.shape
        preds = preds.view(B, -1)
        targets = targets.view(B, -1)
        hard_targets = hard_targets.reshape(B, -1)
        # Predicted distribution
        log_preds = F.log_softmax(preds, dim=1)

        '''preds = F.sigmoid(preds)
        pred_sum = preds.mean(dim=(1,2))
        target_sum = targets.mean(dim=(1,2))
        sparsity = F.mse_loss(pred_sum, target_sum)
        w = 1.0 + alpha * (targets > 0.1).float()
        mse_loss = ((preds - targets)**2 * w).mean(dim=(2,1,0))'''

        kl_div = SeqMatcher.kl_loss(log_preds, targets)
        aux = SeqMatcher.auxiliary_cov_entropy_loss(log_preds.exp(), hard_targets)
        return kl_div + aux, kl_div, aux

    '''@staticmethod
    def loss_fn(paln_map, naln_map, soft_pref_map, soft_nref_map, hard_pref_map, hard_nref_map, pos_score, neg_score, margin=1, use_ranking=True):
        paln_loss, pbce, pdice = SeqMatcher.soft_heatmap_loss(paln_map, soft_pref_map, hard_pref_map)
        naln_loss, nbce, ndice = SeqMatcher.soft_heatmap_loss(naln_map, soft_nref_map, hard_nref_map)
        aln_loss = paln_loss + naln_loss
        target = torch.ones_like(pos_score)
        ranking_loss = F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)
        #ranking_loss = F.binary_cross_entropy_with_logits((pos_score - neg_score) - margin, target)
        if use_ranking:
            loss = aln_loss + ranking_loss
        else:
            loss = aln_loss
        return loss, aln_loss, ranking_loss, (pbce + nbce)/2, (pdice + ndice)/2'''

    @staticmethod
    def loss_fn(alns, refalns, qp_mat, qn_mat, qpref, qnref, margin=1, use_ranking=True):
        B, C, L = alns.shape
        assert(C == 4 and L == max_length)
        alns = alns.view((B*C, L))
        refalns = refalns.to(dtype=alns.dtype,device=alns.device).view((B*C, L))
        log_probs = F.log_softmax(alns, dim=1)      # (N,L)
        aln_loss = F.kl_div(log_probs, refalns, reduction='batchmean')  # KL(target || pred)
        probs = log_probs.exp().view((B,C,L))
        match_loss = F.mse_loss(probs[:,0,:].sum(dim=1), probs[:,1,:].sum(dim=1)) + F.mse_loss(probs[:,2,:].sum(dim=1), probs[:,3,:].sum(dim=1))
        #bce = torch.tensor(0.0)
        dice = torch.tensor(0.0)
        '''aln_loss, bce, dice = SeqMatcher.bce_dice(alns, refalns)'''
        pos_score = (qp_mat.squeeze(1) * qpref).sum(dim=(1,2))
        neg_score = (qn_mat.squeeze(1) * qnref).sum(dim=(1,2))
        target = torch.ones_like(pos_score)
        ranking_loss = F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)
        if use_ranking:
            loss = aln_loss + ranking_loss
        else:
            loss = aln_loss
        return loss, aln_loss, ranking_loss, pos_score, neg_score, match_loss, dice

class ColMatcher(nn.Module):
    def __init__(self,
            dim=48,
            vocab_size=len(tokens),
            conv_kernel=3,
            dropout=0.1,
            n_layers=0,
            n_heads=4,
            coord_channels=6,
            coord_kernel=5,
            max_len=max_length
        ):
        super().__init__()
        self.colbert = ProtConvColBERT(
            dim=dim,
            vocab_size=vocab_size,
            conv_kernel=conv_kernel,
            dropout=dropout,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len
        )

        self.padding = 0 #conv_kernel//2
        self.coord_regressor = CoordRegressor(
            hidden_channels=coord_channels,
            conv_kernel=coord_kernel
        )
        self.scale = nn.Parameter(torch.tensor(1.))

    def make_differentiable_mask(self, heatmaps, a=10, temp=0.05):
        H, W = max_length, max_length
        coords = CoordRegressor.map_to_coords(heatmaps)
        B = coords.size(0)
        start_xy = coords[:, 0, :]
        end_xy = coords[:, 1, :]

        # Convert normalized [0,1] coords to pixel indices
        x0 = coords[:, 0, 0] * (W - 1)
        y0 = coords[:, 0, 1] * (H - 1)
        x1 = coords[:, 1, 0] * (W - 1)
        y1 = coords[:, 1, 1] * (H - 1)

        # Create pixel grids (H, W)
        ys = torch.linspace(0, H - 1, H, device=start_xy.device).view(1, H, 1)
        xs = torch.linspace(0, W - 1, W, device=start_xy.device).view(1, 1, W)

        # Expand to (B, H, W)
        xs = xs.expand(B, -1, -1)
        ys = ys.expand(B, -1, -1)

        # Expand coords to (B, H, W)
        x0 = x0.view(B, 1, 1)
        x1 = x1.view(B, 1, 1)
        y0 = y0.view(B, 1, 1)
        y1 = y1.view(B, 1, 1)

        # Compute sigmoid-based edges
        mask_x = mask_sigmoid(xs - x0,temp=temp,a=a) * mask_sigmoid(x1 - xs,temp=temp,a=a)
        mask_y = mask_sigmoid(ys - y0,temp=temp,a=a) * mask_sigmoid(y1 - ys,temp=temp,a=a)
        mask = mask_x * mask_y  # (B, H, W)
        return mask

    def forward(self, seq, seq_attnmat, seq_padmask, temp=0.1, a=1):
        seqemb, seqdel = self.colbert(seq, seq_attnmat, seq_padmask)
        qp_mat = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,1,:,:], seq_padmask[:,0,:], seq_padmask[:,1,:]).unsqueeze(1)
        qn_mat = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,2,:,:], seq_padmask[:,0,:], seq_padmask[:,2,:]).unsqueeze(1)
        pgmap = self.coord_regressor(qp_mat)
        ngmap = self.coord_regressor(qn_mat)
        qp_mask = self.make_differentiable_mask(pgmap, temp=temp, a=a)
        qn_mask = self.make_differentiable_mask(ngmap, temp=temp, a=a)
        qp_mat_masked = qp_mat * qp_mask.unsqueeze(1) * self.scale
        qn_mat_masked = qn_mat * qn_mask.unsqueeze(1) * self.scale
        return torch.cat((qp_mat, qn_mat), dim=1), \
            torch.cat((qp_mat_masked, qn_mat_masked), dim=1), \
                torch.cat((pgmap, ngmap), dim=1) #(B, 2, H, W), (B, 2, H, W), (B, 4, H, W)

    @staticmethod
    def loss_fn(mat_masked, hmap, hmapref, margin=0.5, locality_coef=0.01, temp=0.2, use_ranking=False, mask_padding=0):
        def score_matrix(sim):
            max_sim, _ = sim.max(dim=2)
            probs = F.softmax(sim, dim=2)         # shape: (B, Q, D)
            positions = torch.arange(max_length).to(sim.device)
            expected_idx = (probs * positions).sum(dim=2)  # shape: (B, Q)
            return max_sim.sum(dim=1), expected_idx

        pos_score, pos_idx = score_matrix(mat_masked[:,0,:,:])
        neg_score, neg_idx = score_matrix(mat_masked[:,1,:,:])
        coord_loss = CoordRegressor.loss_fn(hmap, hmapref)
        coords = CoordRegressor.map_to_coords(hmap) #(B, 4, 2)

        pstart_xy = coords[:, 0, :]
        pend_xy = coords[:, 1, :]
        nstart_xy = coords[:, 2, :]
        nend_xy = coords[:, 3, :]

        qpspan = torch.stack((pstart_xy[:,0],pend_xy[:,0]),dim=1)
        qnspan = torch.stack((nstart_xy[:,0],nend_xy[:,0]),dim=1)
        pos_order_loss = ProtConvColBERT.singular_locality_loss(pos_idx, qpspan, temp=temp, padding=mask_padding)
        neg_order_loss = ProtConvColBERT.singular_locality_loss(neg_idx, qnspan, temp=temp, padding=mask_padding)
        order_loss = pos_order_loss + neg_order_loss
        target = torch.ones_like(pos_score)
        ranking_loss = F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)
        #ranking_loss = F.binary_cross_entropy_with_logits((pos_score - neg_score) - margin, target)
        if use_ranking:
            aux = locality_coef * order_loss + coord_loss
            loss = aux + ranking_loss
        else:
            aux = coord_loss
            loss = aux
        return loss, aux, ranking_loss, pos_score, neg_score, coord_loss, order_loss

class CoordMatcher(nn.Module):
    def __init__(self,
            dim=48,
            vocab_size=len(tokens),
            conv_kernel=3,
            dropout=0.1,
            n_layers=0,
            n_heads=4,
            coord_channels=2,
            coord_layers=3,
            max_len=max_length
        ):
        super().__init__()
        self.colbert = ProtConvColBERT(
            dim=dim,
            vocab_size=vocab_size,
            conv_kernel=conv_kernel,
            dropout=dropout,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len
        )

        self.padding = 0 #conv_kernel//2
        self.coord_regressor = CoordRegressorUNet(
            base=coord_channels,
            depth=coord_layers
        )
        self.scale = nn.Parameter(torch.tensor(1.))

    def make_differentiable_mask(self, coords, qlen, dlen, a=10, temp=0.05):
        B = coords.size(0)
        H, W = max_length, max_length

        # Convert normalized [0,1] coords to pixel indices
        x0 = coords[:, 0]
        y0 = coords[:, 1]
        x1 = dlen - coords[:, 2] - 1
        y1 = qlen - coords[:, 3] - 1

        # Create pixel grids (H, W)
        ys = torch.linspace(0, H - 1, H, device=coords.device).view(1, H, 1)
        xs = torch.linspace(0, W - 1, W, device=coords.device).view(1, 1, W)

        # Expand to (B, H, W)
        xs = xs.expand(B, -1, -1)
        ys = ys.expand(B, -1, -1)

        # Expand coords to (B, H, W)
        x0 = x0.view(B, 1, 1)
        x1 = x1.view(B, 1, 1)
        y0 = y0.view(B, 1, 1)
        y1 = y1.view(B, 1, 1)

        # Compute sigmoid-based edges
        mask_x = mask_sigmoid(xs - x0,temp=temp,a=a) * mask_sigmoid(x1 - xs,temp=temp,a=a)
        mask_y = mask_sigmoid(ys - y0,temp=temp,a=a) * mask_sigmoid(y1 - ys,temp=temp,a=a)
        mask = mask_x * mask_y  # (B, H, W)
        return mask

    def forward(self, seq, seq_attnmat, seq_padmask, seqlen, temp=0.2, a=1):
        seqemb, seqdel = self.colbert(seq, seq_attnmat, seq_padmask)
        qp_mat = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,1,:,:], seq_padmask[:,0,:], seq_padmask[:,1,:]).unsqueeze(1)
        qn_mat = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,2,:,:], seq_padmask[:,0,:], seq_padmask[:,2,:]).unsqueeze(1)
        qlen = seqlen[:,0]
        plen = seqlen[:,1]
        nlen = seqlen[:,2]
        qp_coords = self.coord_regressor(qp_mat, qlen, plen)
        qn_coords = self.coord_regressor(qn_mat, qlen, nlen)
        qp_coords_dummy = torch.zeros_like(qp_coords)
        qn_coords_dummy = torch.zeros_like(qn_coords)
        qp_mask = self.make_differentiable_mask(qp_coords_dummy, qlen=qlen, dlen=plen, temp=temp, a=a).unsqueeze(1)
        qn_mask = self.make_differentiable_mask(qn_coords_dummy, qlen=qlen, dlen=nlen, temp=temp, a=a).unsqueeze(1)
        qp_mat_masked = qp_mat * qp_mask * self.scale
        qn_mat_masked = qn_mat * qn_mask * self.scale
        return torch.cat((qp_mat, qn_mat), dim=1), \
            torch.cat((qp_mat_masked, qn_mat_masked), dim=1), \
                torch.cat((qp_mask, qn_mask), dim=1), \
                    torch.stack((qp_coords, qn_coords),dim=1) #(B, 2, H, W), (B, 2, H, W), (B, 2, H, W), (B, 2, 4)

    @staticmethod
    def loss_fn(mat_masked, coords, coords_ref, seqlen, margin=0.5, locality_coef=0.01, temp=0.2, use_ranking=False, mask_padding=0):
        def score_matrix(sim):
            max_sim, _ = sim.max(dim=2)
            probs = F.softmax(sim, dim=2)         # shape: (B, Q, D)
            positions = torch.arange(max_length).to(sim.device)
            expected_idx = (probs * positions).sum(dim=2)  # shape: (B, Q)
            return max_sim.sum(dim=1), expected_idx
        qlen = seqlen[:, 0]
        coord_loss = CoordRegressorUNet.loss_fn(coords, coords_ref)
        pos_score, pos_idx = score_matrix(mat_masked[:,0,:,:])
        neg_score, neg_idx = score_matrix(mat_masked[:,1,:,:])

        pstart_xy = coords[:, 0, :2]
        pend_xy = coords[:, 0, 2:4]
        nstart_xy = coords[:, 1, :2]
        nend_xy = coords[:, 1, 2:4]

        qpspan = torch.stack((pstart_xy[:,0],qlen - pend_xy[:,0] - 1),dim=1)
        qnspan = torch.stack((nstart_xy[:,0],qlen - nend_xy[:,0] - 1),dim=1)
        pos_order_loss = ProtConvColBERT.singular_locality_loss(pos_idx, qpspan, temp=temp, padding=mask_padding)
        neg_order_loss = ProtConvColBERT.singular_locality_loss(neg_idx, qnspan, temp=temp, padding=mask_padding)
        order_loss = pos_order_loss + neg_order_loss
        target = torch.ones_like(pos_score)
        ranking_loss = F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)
        #ranking_loss = F.binary_cross_entropy_with_logits((pos_score - neg_score) - margin, target)
        if use_ranking:
            aux = locality_coef * order_loss
            loss = aux + ranking_loss
        else:
            aux = torch.tensor(0.0)
            loss = aux
        return loss, aux, ranking_loss, pos_score, neg_score, coord_loss, order_loss

class ColBERT(nn.Module):
    def __init__(self,
            dim=48,
            vocab_size=len(tokens),
            conv_kernel=3,
            dropout=0.1,
            n_layers=0,
            n_heads=4,
            max_len=max_length,
            enc_type=0,
            symmetric=False,
            init_token: Optional[torch.Tensor]=None
        ):
        super().__init__()
        self.colbert = ProtConvColBERT(
            dim=dim,
            vocab_size=vocab_size,
            conv_kernel=conv_kernel,
            dropout=dropout,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
            enc_type=enc_type,
            symmetric=symmetric,
            init_token=init_token
        )
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.gap_f = nn.Parameter(torch.tensor(0.05))

    def forward(self, seq, seq_attnmat, seq_padmask, seqlen, temp=0.2, a=1):
        seqemb, blosum_mask = self.colbert(seq, seq_attnmat, seq_padmask)
        qp_mat = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,1,:,:], seq_padmask[:,0,:], seq_padmask[:,1,:]).unsqueeze(1)
        qn_mat = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,2,:,:], seq_padmask[:,0,:], seq_padmask[:,2,:]).unsqueeze(1)
        mats = torch.cat((qp_mat, qn_mat), dim=1)
        #mats_masked = mats * blosum_mask * self.scale
        return mats, blosum_mask #(B, 2, H, W), (B, 2, H, W), (B, 2, H, W), (B, 2, 4)

    ''''@staticmethod
    def loss_fn(mat_masked, seqlen, seq_padmask, margin=0.5, locality_coef=2, temp=0.2, use_ranking=False, mask_padding=0):
        def score_matrix(mat, qmask, dmask, temp=0.08):
            def score_m(sim, seqmask, softplus_b=3.5):
                #max_sim, _ = sim.max(dim=2)
                max_sim = temp*torch.logsumexp(sim/temp, dim=2)
                #max_sim = max_sim * (1 - seqmask.to(dtype=sim.dtype))
                max_sim = max_sim * seqmask.to(dtype=sim.dtype)
                probs = F.softmax(sim, dim=2)         # shape: (B, Q, D)
                positions = torch.arange(max_length).to(sim.device)
                expected_idx = (probs * positions).sum(dim=2)  # shape: (B, Q)
                return max_sim.sum(dim=1), expected_idx
            qscore, q_idx = score_m(mat, qmask)
            dscore, d_idx = score_m(mat.transpose(1,2), dmask)
            return qscore/2 + dscore/2, q_idx, d_idx
        B = seqlen.size(0)
        qlen = seqlen[:, 0]
        plen = seqlen[:, 1]
        nlen = seqlen[:, 2]
        coord_loss = torch.tensor(0.0)
        coords = torch.zeros((B),dtype=mat_masked.dtype,device=mat_masked.device)
        pos_score, qpos_idx, dpos_idx = score_matrix(mat_masked[:,0,:,:], seq_padmask[:,0,:], seq_padmask[:,1,:])
        neg_score, qneg_idx, dneg_idx = score_matrix(mat_masked[:,1,:,:], seq_padmask[:,0,:], seq_padmask[:,2,:])
        qspan = torch.stack((coords,qlen - coords - 1),dim=1)
        pspan = torch.stack((coords,plen - coords - 1),dim=1)
        nspan = torch.stack((coords,nlen - coords - 1),dim=1)
        pos_order_loss = ProtConvColBERT.singular_locality_loss(qpos_idx, qspan, temp=temp, padding=mask_padding) + ProtConvColBERT.singular_locality_loss(dpos_idx, pspan, temp=temp, padding=mask_padding)
        neg_order_loss = ProtConvColBERT.singular_locality_loss(qneg_idx, qspan, temp=temp, padding=mask_padding) + ProtConvColBERT.singular_locality_loss(qneg_idx, nspan, temp=temp, padding=mask_padding)
        order_loss = pos_order_loss + neg_order_loss
        target = torch.ones_like(pos_score)
        ranking_loss = F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)
        aux = locality_coef * order_loss
        loss = ranking_loss
        return loss, aux, ranking_loss, pos_score, neg_score, coord_loss, order_loss'''

    def loss_fn(self, mats, blosum_mask, spanref, seqmatchmask, seqspanmask, pairmatchmask,  with_gap=False, locality_coef=1, margin=0.5, temp=0.08, use_ranking=False, mask_padding=0):
        def score_matrix(mats, match_mask, span_mask, temp=0.1, with_gap=False):
            def score_gap(spanmask, matchmask):
                gapmask = spanmask.half() - matchmask.half()
                B, L = gapmask.shape
                shifted = torch.cat((torch.zeros((B,1), dtype=gapmask.dtype, device=gapmask.device), gapmask[:,:-1]),dim=1)
                gapstart = (gapmask == 1) & (shifted == 0)
                return self.gap_f * (gapmask.sum(dim=1) + 10*gapstart.sum(dim=1))

            def score_m(sim, matchmask):
                max_sim, _ = sim.max(dim=2)
                #max_sim = temp*torch.logsumexp(sim/temp, dim=2)
                #max_sim = max_sim * (1 - seqmask.to(dtype=sim.dtype))
                max_sim = max_sim * matchmask.to(dtype=sim.dtype)
                match_score = max_sim.sum(dim=1)
                return match_score
            qmatch = score_m(mats, match_mask[:,0,:])
            dmatch = score_m(mats.transpose(1,2), match_mask[:,1,:])
            qscore = qmatch - (score_gap(span_mask[:,0,:], match_mask[:,0,:]) if with_gap else 0.0)
            dscore = dmatch - (score_gap(span_mask[:,1,:], match_mask[:,1,:]) if with_gap else 0.0)
            return qscore/2 + dscore/2
        
        def match_loss(mats, pair_match_mask, temp = 0.1):
            def match(sim, matchmask):
                expsim = (torch.exp((sim * matchmask)/temp)).sum(dim=2)
                sumexpsim = (torch.exp(sim/temp)).sum(dim=2)
                return (1 - torch.div(expsim,sumexpsim)).mean(dim=1) + (1-(sim * matchmask).sum(dim=2).mean(dim=1))
            B, C, L, _ = mats.shape
            assert(mats.shape[3] == L and C == 2)
            mat_reshaped = mats.view((B*C,L,L))
            pair_match_mask_reshaped = pair_match_mask.view((B*C, L, L))
            qmatch = match(mat_reshaped, pair_match_mask_reshaped)
            dmatch = match(mat_reshaped.transpose(1,2), pair_match_mask_reshaped.transpose(1,2))
            m_loss = (qmatch/2 + dmatch/2).mean(dim=0)
            #print(qmatch.shape, dmatch.shape, m_loss.shape)
            return m_loss

        mats_masked = mats * blosum_mask
        coord_loss = match_loss(mats, pairmatchmask, temp=1.3)
        #print(coord_loss)
        B = mats_masked.size(0)
        positions = torch.arange(max_length).to(mats.device)
        def get_expected_idx(m):
            probs = F.softmax(m, dim=2)
            return (probs * positions).sum(dim=2)
        qp_idx = get_expected_idx(mats[:,0,:,:])
        p_idx = get_expected_idx(mats[:,0,:,:].transpose(1,2))
        qn_idx = get_expected_idx(mats[:,1,:,:])
        n_idx = get_expected_idx(mats[:,1,:,:].transpose(1,2))

        qp_order_loss = ProtConvColBERT.singular_locality_loss(qp_idx, spanref[:,0,:], temp=temp, padding=mask_padding)
        p_order_loss = ProtConvColBERT.singular_locality_loss(p_idx, spanref[:,1,:], temp=temp, padding=mask_padding)
        qn_order_loss = ProtConvColBERT.singular_locality_loss(qn_idx, spanref[:,2,:], temp=temp, padding=mask_padding)
        n_order_loss = ProtConvColBERT.singular_locality_loss(n_idx, spanref[:,3,:], temp=temp, padding=mask_padding)

        pos_score = score_matrix(mats_masked[:,0,:,:], seqmatchmask[:,0:2,:], seqspanmask[:,0:2,:], with_gap=with_gap)
        neg_score = score_matrix(mats_masked[:,1,:,:], seqmatchmask[:,2:4,:], seqspanmask[:,2:4,:], with_gap=with_gap)

        order_loss = qp_order_loss + p_order_loss + qn_order_loss + n_order_loss
        target = torch.ones_like(pos_score)
        ranking_loss = F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)
        aux = locality_coef * order_loss
        loss = coord_loss + (ranking_loss if use_ranking else 0.0)
        return loss, aux, ranking_loss, pos_score, neg_score, coord_loss, order_loss

class ColBERT_direct(nn.Module):
    def __init__(self,
            dim=48,
            vocab_size=len(tokens),
            conv_kernel=3,
            dropout=0.1,
            n_layers=0,
            n_heads=4,
            max_len=max_length,
            enc_type=0,
            symmetric=False,
            init_token: Optional[torch.Tensor]=None,
            guide_blosum: Optional[torch.Tensor]=None,
            normalized=True
        ):
        super().__init__()
        self.colbert = ProtConvColBERT(
            dim=dim,
            vocab_size=vocab_size,
            conv_kernel=conv_kernel,
            dropout=dropout,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
            enc_type=enc_type,
            symmetric=symmetric,
            init_token=init_token,
            normalized=normalized
        )
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.gap_f = nn.Parameter(torch.tensor(2.))
        self.simloss_scale = nn.Parameter(torch.tensor(5.0))
        self.guide_blosum = guide_blosum

    def forward(self, seq, seq_attnmat, seq_padmask, seqlen):
        seqemb, blosum_mask = self.colbert(seq, seq_attnmat, seq_padmask)
        mats = self.colbert.compute_sim_mat(seqemb[:,0,:,:], seqemb[:,1,:,:], seq_padmask[:,0,:], seq_padmask[:,1,:]).unsqueeze(1)
        #mats_masked = mats * blosum_mask * self.scale
        return mats, blosum_mask #(B, 2, H, W), (B, 2, H, W), (B, 2, H, W), (B, 2, 4)

    @torch.jit.export
    def encode(self, seq):
        return self.colbert.encode(seq)

    def sim_loss(self, device):
        if self.guide_blosum is None:
            return torch.tensor(0.0)
        target = self.guide_blosum.to(device)
        e = self.colbert.encoder.embedding[0].weight.to(device)
        if self.colbert.encoder.normalized:
            normalized = F.normalize(e, p=2., dim=-1)
        else:
            normalized = e
        aa_sim = normalized @ normalized.T           # (24,24)
        return - (aa_sim * target).mean()

    def score_gap(self, spanmask, matchmask):
        gapmask = spanmask.half() - matchmask.half()
        B, L = gapmask.shape
        shifted = torch.cat((torch.zeros((B,1), dtype=gapmask.dtype, device=gapmask.device), gapmask[:,:-1]),dim=1)
        gapstart = (gapmask == 1) & (shifted == 0)
        return self.gap_f * (gapmask.sum(dim=1) + 10*gapstart.sum(dim=1))

    def score_m(self, sim, sim_blosum, pmatchmask, t = 0.01):
        #max_sim, _ = sim.max(dim=2)
        #weights = F.softmax(sim / t, dim=2)  # shape: (batch_size, num_docs)

        # Weighted sum over score
        #max_sim = (sim_blosum * weights).sum(dim=2)

        #max_sim = temp*torch.logsumexp(sim/temp, dim=2)
        #max_sim = max_sim * (1 - seqmask.to(dtype=sim.dtype))
        #print(sim_blosum.shape, pmatchmask.shape)
        sim_blosum *= pmatchmask.to(dtype=sim_blosum.dtype).view(sim_blosum.shape)
        match_score = sim_blosum.sum(dim=(1,2))
        return match_score

    def score_matrix(self, mat, mat_blosum, match_mask, span_mask, pair_match_mask, temp=0.1, with_gap=False):
        qmatch = self.score_m(mat, mat_blosum, pair_match_mask)
        dmatch = self.score_m(mat.transpose(1,2), mat_blosum.transpose(1,2), pair_match_mask.transpose(1,2))
        qscore = qmatch - (self.score_gap(span_mask[:,0,:], match_mask[:,0,:]) if with_gap else 0.0)
        dscore = dmatch - (self.score_gap(span_mask[:,1,:], match_mask[:,1,:]) if with_gap else 0.0)
        return qscore/2 + dscore/2

    def mmatch(self, sim, matchmask, nummatch):
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

    def match_loss(self, mats, pair_match_mask, temp = 0.1):
        B, C, L, _ = mats.shape
        assert(mats.shape[3] == L and C == 1)
        mat_reshaped = mats.view((B*C,L,L))
        pair_match_mask_reshaped = pair_match_mask.view((B*C, L, L))
        nummatch = pair_match_mask_reshaped.sum(dim=(1,2))
        qmatch = self.mmatch(mat_reshaped, pair_match_mask_reshaped, nummatch)
        dmatch = self.mmatch(mat_reshaped.transpose(1,2), pair_match_mask_reshaped.transpose(1,2), nummatch)
        m_loss = (qmatch/2 + dmatch/2).mean(dim=0)
        return m_loss

    def smatch(self, sim, matchmask, nummatch):
        selection = F.relu((sim * matchmask).sum(dim=2))
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
        positions = torch.arange(max_length).to(mats.device)
        return (probs * positions).sum(dim=2)

    def loss_fn(self, mats, blosum_mask, spanref, seqmatchmask, seqspanmask, pairmatchmask, target_score, with_gap=False, use_scoring=False, locality_coef=1, temp=0.08, mask_padding=0, guide_scale=0.001):
        #print(pairmatchmask.shape)
        mats_masked = mats * blosum_mask
        coord_loss = self.match_loss(mats, pairmatchmask, temp=temp)
        #print(coord_loss)
        B = mats_masked.size(0)
        q_idx = self.get_expected_idx(mats[:,0,:,:])
        d_idx = self.get_expected_idx(mats[:,0,:,:].transpose(1,2))
        q_order_loss = ProtConvColBERT.singular_locality_loss(q_idx, spanref[:,0,:], temp=0.08, padding=mask_padding)
        d_order_loss = ProtConvColBERT.singular_locality_loss(d_idx, spanref[:,1,:], temp=0.08, padding=mask_padding)

        aln_score = self.score_matrix(mats[:,0,:,:], mats_masked[:,0,:,:], seqmatchmask[:,0:2,:], seqspanmask[:,0:2,:], pairmatchmask, with_gap=with_gap)
        #score_loss = F.mse_loss(aln_score, target_score)
        score_loss = torch.square(torch.ones_like(aln_score) - torch.div(aln_score,target_score)).mean()
        #score_loss = torch.div(torch.clamp(aln_score,min=0.01)+1,target_score+1)
        #score_loss = torch.square(torch.log(score_loss)).mean()
        order_loss = q_order_loss + d_order_loss
        aux = locality_coef * order_loss
        sloss = self.sim_loss(device=mats.device)
        loss = coord_loss + guide_scale * sloss
        if use_scoring:
            loss += score_loss
        return loss, aux, score_loss, aln_score, coord_loss, self.signal_loss(mats, pairmatchmask), sloss
