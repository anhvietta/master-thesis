import jax
from jax2torch import jax2torch
import torch
import torch.nn.functional as F
from constants import blosum62_gttl, AMINO_ACIDS_GTTL
from ssw import sw_affine
from sw import sw_affine as sw
import matplotlib.pyplot as plt

alphasize = len(AMINO_ACIDS_GTTL)

B = 1
l = 100
l1 = 80
l2 = 70
gap_ext = -1.
gap_open = -11.
temp = 1.

def scores(s1, s2, matrix):
    B, L = s1.shape
    s1 = s1.to(dtype=torch.int64)
    s2 = s2.to(dtype=torch.int64)
    rows = torch.index_select(matrix, 0, s1.reshape(-1)).reshape(B, L, -1)
    s2_exp = s2.unsqueeze(1).expand(-1, s1.size(1), -1)
    scores = torch.gather(rows, 2, s2_exp)
    return scores

def set_equal(s1, s2, l, h):
    s2[:, l:h] = s1[:, l:h]

mask = (torch.arange(l) < l1)[:,None] * (torch.arange(l) < l2)[None,:]
mask = mask.unsqueeze(0).to(dtype=torch.float32).expand(B, -1, -1).contiguous().cuda()
s1 = F.pad(
    torch.randint(low=0, high=alphasize-2, size=(1,l1)),
    (0,l-l1),
    value=alphasize-1
)
s2 = F.pad(
    torch.randint(low=0, high=alphasize-2, size=(1,l2)),
    (0,l-l2),
    value=alphasize-1
)
blosum62 = torch.tensor(blosum62_gttl, dtype=torch.float32, requires_grad=True)
score_mats = scores(s1,s2,blosum62).expand(B, -1, -1).contiguous().cuda()
print(score_mats.requires_grad)

f = jax2torch(jax.jit(
    sw_affine(
        restrict_turns=False,
        penalize_turns=False,
        NINF=-1e8,
        temp=temp,
        eps=1e-8
    )
))
result = f(score_mats, mask, gap_ext, gap_open).detach().cpu()
print(result.min(), result.max())
#plt.imsave('/scratch/stud2018/ata/ssw_test.png', torch.log(result + 1e-8)[0])
