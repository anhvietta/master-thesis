import jax
import jax.numpy as jnp
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/10.1.243-fasrc01/"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def sw_affine(restrict_turns=True,
             penalize_turns=True,
             batch=True, unroll=2, NINF=-1e30, eps=1e-8):
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
    def sco(x, padmask, gap=0.0, open=0.0, temp=1.0):

        def _soft_maximum(x, axis=None, mask=None):
            def _logsumexp(y):
                y = jnp.maximum(y,NINF)
                if mask is None: return jax.nn.logsumexp(y, axis=axis)
                else: return y.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(y - y.max(axis, keepdims=True)), axis=axis) + eps)
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
        x = x + padmask * NINF
        mask = 1 - padmask

        sm, prev, idx = rotate(x[:-1,:-1])
        hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]

        # sink
        return _soft_maximum(hij + x[1:,1:,None], mask=mask[1:,1:,None])
    # traceback to get alignment (aka. get marginals)
    traceback = jax.grad(sco)

    # add batch dimension
    if batch: return jax.vmap(traceback,(0,0,None,None,None))
    else: return traceback
