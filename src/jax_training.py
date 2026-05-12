import os
import random
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
from flax import traverse_util
import optax
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import argparse as ap
from dataset import JaxDataset
from jax_encoder import JaxEncoder, loss_fn, sim_loss_fn
from constants import max_length, input_dim, latent_dim, hidden_dim, TOKENS, attention_mask_window_size, PAD_TOKEN, blosum62_gttl
from utils import tokens, pad_token
from mds import get_mdsdecmp
import numpy as np
import cv2
import pickle
#769228, 585305, 699545, 388597
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
base_data = '/scratch/stud2018/ata/ma_data/'
base_ckpts = '/scratch/stud2018/ata/'
base_out = '/scratch/stud2018/ata/'
outfile = "out6.csv"
#random.seed(42)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

def save_checkpoint(params, optimizer_state, epoch, global_step, path="checkpoint.pkl"):
    """
    Save model parameters, optimizer state, and training info.
    """
    checkpoint = {
        "params": params,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "global_step": global_step
    }
    # Serialize PyTrees
    with open(path, "wb") as f:
        pickle.dump(flax.serialization.to_state_dict(checkpoint), f)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(path="checkpoint.pkl"):
    """
    Load model params, optimizer state, epoch, and global step.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, "rb") as f:
        state_dict = pickle.load(f)

    checkpoint = flax.serialization.from_state_dict({}, state_dict)
    params = checkpoint["params"]
    optimizer_state = checkpoint["optimizer_state"]
    epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]

    print(f"Checkpoint loaded from {path}, resuming from epoch {epoch}, step {global_step}")
    return params, optimizer_state, epoch, global_step

def get_decay_factor(start,end,duration):
    return (end/start) ** (1/duration)

def make_ref_map(alnref):
    qaln = alnref[:, 0, :]
    daln = alnref[:, 1, :]
    B, n = qaln.shape

    # Initialize empty boolean map
    m = jnp.zeros((B, n, n), dtype=bool)

    # Get indices where qaln and daln are True
    q_b, q_i = jnp.nonzero(qaln)  # batch index, query index
    d_b, d_i = jnp.nonzero(daln)  # batch index, target index
    
    # Ensure batch indices match
    assert jnp.all(q_b == d_b), "Batch indices do not match"

    # Scatter True into m
    m = m.at[q_b, q_i, d_i].set(True)
    return m

"""def make_ref_map_single(alnref):
    qaln = alnref[0, :]
    daln = alnref[1, :]
    q_pos = jnp.where(qaln)[0]  # positions of True in qaln
    d_pos = jnp.where(daln)[0]  # positions of True in daln
    n_pairs = q_pos.shape[0]

    # Only consider up to min number of Trues
    n_pairs = jnp.minimum(n_pairs, d_pos.shape[0])
    q_pos = q_pos[:n_pairs]
    d_pos = d_pos[:n_pairs]

    # Create a zero matrix
    L = qaln.shape[0]
    ref_map = jnp.zeros((L, L), dtype=bool)

    # Use index update with scatter
    ref_map = ref_map.at[q_pos, d_pos].set(True)
    return ref_map

# vmap over batch
def make_ref_map(alnref):
    return jax.vmap(make_ref_map_single)(alnref)"""

def get_model_info_str(model_params, l=['!'], depth=0):
    l = ["-","+","=",'*',"#"]
    def make_str(attr, depth=0):
        if depth >= len(l):
            raise RuntimeError("Out of depth")
        if isinstance(attr, list):
            return l[depth].join([str(a) if not isinstance(a,list) else make_str(a, depth+1) for a in attr])
        else:
            return str(attr)
    return "_".join([
        make_str(model_params["dims"]),
        make_str(model_params["conv_kernels"]),
        make_str(model_params["dilations"]),
        make_str(model_params["outdim"])
    ])

def make_span_mask(span):
    """
    span: shape (B, 2, 2)  -> [batch, query/target, start/end]
    returns: mask of shape (B, 2, maxlen)
    """
    B, C, D = span.shape
    assert C == 2 and D == 2, "span must have shape (B,2,2)"

    starts = span[:, :, 0:1]  # shape (B, 2, 1)
    ends   = span[:, :, 1:2]  # shape (B, 2, 1)

    idx = jnp.arange(max_length).reshape(1, 1, max_length)  # shape (1, 1, maxlen)

    mask = (idx >= starts) & (idx <= ends)          # broadcast to (B, 2, maxlen)
    return mask

def logger(out_path):
    def logging(msg):
        print(msg)
        with open(out_path, "a") as file:
            file.write(str(msg)+"\n")
    return logging

# Create masks for weight decay
'''def decay_mask(param_name):
    # Exclude bias and LayerNorm weights
    if "bias" in param_name or "LayerNorm" in param_name or "layer_norm" in param_name:
        return False
    return True

# Use tree_map to create masks
def make_mask(params):
    flat = {}
    for path, _ in traverse_dict(params):  # helper function
        flat[path] = decay_mask(path)
    return unflatten_dict(flat)'''

def decay_mask(params):
    """
    Returns a PyTree mask with True for params that should have weight decay,
    and False for bias/LayerNorm weights.
    """
    flat = traverse_util.flatten_dict(params)

    mask_flat = {}
    for path, v in flat.items():
        # path is a tuple like ('tokenizer', 'embedding', 'Embed_0', 'embedding')
        name = path[-1]  # usually the param name
        if "bias" in name.lower() or "layernorm" in name.lower() or "scale" in name.lower():
            mask_flat[path] = False
        else:
            mask_flat[path] = True

    return traverse_util.unflatten_dict(mask_flat)


def jax_batch_generator(dataset, batch_size, indices=None, shuffle=True):
    if not indices:
        indices = np.arange(num_rows)
    num_rows = len(indices)
    num_batches = num_rows // batch_size + (0 if num_rows % batch_size == 0 else 1)
    if shuffle:
        np.random.shuffle(indices)

    for b_idx in range(num_batches):
        start = b_idx * batch_size
        end = min(num_rows, start + batch_size)
        batch_idx = indices[start:end]
        batch_items = [dataset[i] for i in batch_idx]

        # Unpack each field across the batch
        seqs, lengths, spans, alns, scores, idxs = zip(*batch_items)

        # Stack each field individually
        batch_seqs = jnp.stack(seqs)        # shape: (B,2,L)
        batch_lengths = jnp.stack(lengths)  # shape: (B,2)
        batch_spans = jnp.stack(spans)      # shape: (B,2,2)
        batch_alns = jnp.stack(alns)        # shape: (B,2,...)
        batch_scores = jnp.stack(scores)    # shape: (B,1)
        batch_idxs = jnp.array(idxs)        # shape: (B,)

        yield batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs

def has_nan(params):
    return any(jnp.isnan(p).any() for p in jax.tree_util.tree_leaves(params))

def init_grad_accumulator(params):
    return jax.tree_util.tree_map(jnp.zeros_like, params)

@flax.struct.dataclass
class State:
    params: dict
    opt_state: optax.OptState
    grad_acc: dict
    step_counter: int

def train_loop(
    params, optimizer, optimizer_state, dataset, model, forward, loss_fn, grad_acc_iter, global_step, batch_size,
    shuffle=True, logging=print,
    indices=None, max_grad_norm=1.0, temp=0.08, guide_scale=0.001, abs_scale=0.1,
    num_enc_used=1, threshold=0.4, max_scale=0.5
    ):
    total_loss = 0.0
    total_aloss = 0.0
    total_mloss = 0.0
    total_mmloss = 0.0
    total_sigloss = 0.0
    total_simloss = 0.0
    n = len(indices if indices is not None else dataloader)
    num_batch = n // batch_size + (0 if n % batch_size == 0 else 1)

    grad_acc = init_grad_accumulator(params)
    step_counter = 0
    has_guide = model.guide_blosum is not None
    train = True

    @jax.jit
    def _step(state, batch):
        """
        batch: tuple of arrays from generator:
            batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs
        """
        batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs, batch_refmap = batch
        seq_padding_mask = (batch_seqs == tokens.index(pad_token)).astype(bool)
        batch_spanmask = make_span_mask(batch_spans)
        def loss_wrap(params):
            maps, traceback = forward(
                {"params": params},
                batch_seqs,
                seq_padding_mask,
                train
            )
            loss, aln_loss, matchloss, mmatchloss, signalloss = loss_fn(
                maps,
                traceback,
                batch_spans,
                batch_alns,
                batch_spanmask,
                batch_refmap,
                temp,
                abs_scale,
                threshold,
                max_scale
            )
            loss = jnp.mean(loss)
            aln_loss = jnp.mean(aln_loss)
            matchloss = jnp.mean(matchloss)
            mmatchloss = jnp.mean(mmatchloss)
            signalloss = jnp.mean(signalloss)
            sim_loss = sim_loss_fn(
                params["tokenizer"]["embedding"]["embedding"],
                model.guide_blosum,
                model.normalized
            ) * guide_scale if has_guide else 0.
            """jax.debug.print(
                "traceback min={tb_min}, max={tb_max} | maps min={m_min}, max={m_max} | sim_loss={simloss} | mmatch_loss={mmatchloss} | aln_loss={aln_loss} | matchloss={matchloss}",
                tb_min=traceback.min(),
                tb_max=traceback.max(),
                m_min=maps.min(),
                m_max=maps.max(),
                simloss=sim_loss,
                mmatchloss=mmatchloss,
                aln_loss=aln_loss,
                matchloss=matchloss
            )"""
            return loss + sim_loss, (aln_loss, matchloss, mmatchloss, signalloss, sim_loss)

        # Compute gradient
        (loss_value, aux), grads = jax.value_and_grad(loss_wrap, has_aux=True)(state.params)
        grad_acc = jax.tree_util.tree_map(lambda g_acc, g: g_acc + g / grad_acc_iter, state.grad_acc, grads)

        def apply_update(_):
            updates, opt_state = optimizer.update(grad_acc, state.opt_state, params=state.params)
            params = optax.apply_updates(state.params, updates) 
            return State(params, opt_state, init_grad_accumulator(params), state.step_counter+1)

        def skip_update(_):
            return State(state.params, state.opt_state, grad_acc, state.step_counter+1)

        new_state = jax.lax.cond(
            train and (state.step_counter % grad_acc_iter == 0),
            apply_update,
            skip_update,
            operand=None
        )

        return new_state, loss_value, aux
    # Iterate over batches
    for batch_idx, batch in enumerate(jax_batch_generator(dataset, batch_size=batch_size, indices=indices, shuffle=shuffle)):
        batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs = batch
        batch_refmap = make_ref_map(batch_alns)
        batch = (batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs, batch_refmap)
        state = State(params, optimizer_state, grad_acc, batch_idx+1)
        new_state, loss_value, aux_losses = _step(state, batch)
        
        #params, batch_stats, optimizer_state, grad_acc, step_counter = new_state
        params = new_state.params
        optimizer_state = new_state.opt_state
        grad_acc = new_state.grad_acc
        step_counter = new_state.step_counter
        
        if jnp.isnan(loss_value):
            logging("NaN detected in loss!")
        if has_nan(params):
            logging("NaN detected in model parameters!")

        aln_loss, match_loss_val, mmatch_loss, signal_loss_val, sim_loss_val, _ = aux_losses
        total_loss += loss_value
        total_aloss += aln_loss
        total_mloss += match_loss_val
        total_mmloss += mmatch_loss
        total_sigloss += signal_loss_val
        total_simloss += sim_loss_val

        global_step += 1

    metrics = {
        "total": total_loss / num_batch,
        "aln": total_aloss / num_batch,
        "match": total_mloss / num_batch,
        "mmatch": total_mmloss / num_batch,
        "sigloss": total_sigloss / num_batch,
        "simloss": total_simloss / num_batch
    }

    return params, optimizer_state, metrics

def train_loop_pmap(
    params, optimizer, optimizer_state, dataset, model, forward, loss_fn, grad_acc_iter, global_step, batch_size,
    shuffle=True, logging=print,
    indices=None, max_grad_norm=1.0, temp=0.08, guide_scale=0.001, abs_scale=0.1,
    num_enc_used=1, max_scale=0.5, log_interval=5
):
    devices = jax.local_devices()
    num_devices = len(devices)

    # Helper: replicate state to devices
    state = State(
        params=params,
        opt_state=optimizer_state,
        grad_acc=init_grad_accumulator(params),
        step_counter=1
    )

    # Shard batch across devices
    def shard(batch):
        return jax.tree_util.tree_map(
            lambda x: x.reshape((num_devices, -1) + x.shape[1:]),
            batch
        )

    # pmap axis name for collective ops
    axis_name = 'batch'

    has_guide = model.guide_blosum is not None
    train = True

    def _step(state, batch):
        batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs, batch_refmap = batch
        seq_padding_mask = (batch_seqs == tokens.index(pad_token)).astype(bool)
        batch_spanmask = make_span_mask(batch_spans)

        def loss_wrap(params):
            maps, traceback = forward(
                {"params": params},
                batch_seqs,
                seq_padding_mask,
                batch_lengths,
                train
            )
            loss, aln_loss, matchloss, mmatchloss, signalloss = loss_fn(
                maps,
                traceback,
                batch_spans,
                batch_alns,
                batch_spanmask,
                batch_refmap,
                temp,
                abs_scale,
                max_scale
            )
            loss = jnp.mean(loss)
            aln_loss = jnp.mean(aln_loss)
            matchloss = jnp.mean(matchloss)
            mmatchloss = jnp.mean(mmatchloss)
            signalloss = jnp.mean(signalloss)
            sim_loss = sim_loss_fn(
                params["tokenizer"]["embedding"]["embedding"],
                model.guide_blosum,
                model.normalized
            ) * guide_scale if has_guide else 0.
            '''jax.debug.print(
                "traceback min={tb_min:.4f}, max={tb_max:.4f} | maps min={m_min:.4f}, max={m_max:.4f} | sim_loss={simloss:.4f} | mmatch_loss={mmatchloss:.4f} | aln_loss={aln_loss:.4f} | matchloss={matchloss:.4f}",
                tb_min=traceback.min(),
                tb_max=traceback.max(),
                m_min=maps.min(),
                m_max=maps.max(),
                simloss=sim_loss,
                mmatchloss=mmatchloss,
                aln_loss=aln_loss,
                matchloss=matchloss
            )'''
            return loss + sim_loss, (aln_loss, matchloss, mmatchloss, signalloss, sim_loss)

        # Compute gradients
        (loss_value, aux), grads = jax.value_and_grad(loss_wrap, has_aux=True)(state.params)

        # Accumulate gradients per device
        grad_acc = jax.tree_util.tree_map(
            lambda g_acc, g: g_acc + g / grad_acc_iter,
            state.grad_acc, grads
        )

        def apply_update(_):
            # average gradients across devices before update
            grads_mean = jax.lax.pmean(grad_acc, axis_name=axis_name)
            updates, new_opt_state = optimizer.update(grads_mean, state.opt_state, params=state.params)
            new_params = optax.apply_updates(state.params, updates)
            return State(new_params, new_opt_state, init_grad_accumulator(new_params), state.step_counter + 1)

        def skip_update(_):
            return State(state.params, state.opt_state, grad_acc, state.step_counter + 1)

        # Only update after grad_acc_iter steps
        new_state = jax.lax.cond(
            train & (state.step_counter % grad_acc_iter == 0),
            apply_update,
            skip_update,
            operand=None
        )

        return new_state, loss_value, aux
    
    step = jax.pmap(_step, axis_name=axis_name, in_axes=(None, (0, 0, 0, 0, 0, 0, 0)), out_axes=(None, None, None))
    
    # Metrics
    total_loss = 0.0
    total_aloss = 0.0
    total_mloss = 0.0
    total_mmloss = 0.0
    total_sigloss = 0.0
    total_simloss = 0.0

    n = len(indices if indices is not None else dataset)
    num_batch = n // batch_size + (0 if n % batch_size == 0 else 1)
    log_iter = num_batch // log_interval
    for batch_idx, batch in enumerate(jax_batch_generator(dataset, batch_size=batch_size, indices=indices, shuffle=shuffle)):
        batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs = batch
        batch_refmap = make_ref_map(batch_alns)
        batch_full = (batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs, batch_refmap)

        # Shard the batch for devices
        batch_sharded = shard(batch_full)

        # Run parallel step
        state, loss_value, aux_losses = step(state, batch_sharded)

        # Take mean of losses across devices
        loss_value = jnp.mean(loss_value)
        aux_losses = jax.tree_util.tree_map(lambda x: jnp.mean(x), aux_losses)

        #params = jax.tree_util.tree_map(lambda x: x[0], state.params)  # pick from first device
        #batch_stats = jax.tree_util.tree_map(lambda x: x[0], state.batch_stats)
        #optimizer_state = jax.tree_util.tree_map(lambda x: x[0], state.opt_state)
        #grad_acc = jax.tree_util.tree_map(lambda x: x[0], state.grad_acc)
        params = state.params
        optimizer_state = state.opt_state
        grad_acc = state.grad_acc
        step_counter = state.step_counter

        aln_loss, match_loss_val, mmatch_loss, signal_loss_val, sim_loss_val = aux_losses
        total_loss += float(loss_value)
        total_aloss += float(aln_loss)
        total_mloss += float(match_loss_val)
        total_mmloss += float(mmatch_loss)
        total_sigloss += float(signal_loss_val)
        total_simloss += float(sim_loss_val)

        global_step += 1

        if jnp.isnan(loss_value):
            logging("NaN detected in loss!")
            logging(str(aux_losses))
            raise RuntimeError("NaN in loss!")
        if has_nan(params):
            logging("NaN detected in model parameters!")
            logging(str(params))
            raise RuntimeError("NaN in model!")
        if (log_iter > 0 and (batch_idx + 1) % log_iter == 0) or log_iter == 0:
            logging(f"{batch_idx+1}/{num_batch}\t{(total_loss / (batch_idx+1)):.2f}\t{(total_mloss / (batch_idx+1)):.2f}")

    metrics = {
        "total": total_loss / num_batch,
        "aln": total_aloss / num_batch,
        "match": total_mloss / num_batch,
        "mmatch": total_mmloss / num_batch,
        "sigloss": total_sigloss / num_batch,
        "simloss": total_simloss / num_batch
    }

    return params, optimizer_state, metrics, global_step


def test_loop(
    params, optimizer_state, dataset, model, forward, loss_fn, grad_acc_iter, global_step, batch_size,
    shuffle=True, logging=print,
    indices=None, max_grad_norm=1.0, temp=0.08, guide_scale=0.001, abs_scale=0.1,
    num_enc_used=1, threshold=0.4, max_scale=0.5
    ):
    total_loss = 0.0
    total_aloss = 0.0
    total_mloss = 0.0
    total_mmloss = 0.0
    total_sigloss = 0.0
    total_simloss = 0.0
    n = len(indices if indices is not None else dataloader)
    num_batch = n // batch_size + (0 if n % batch_size == 0 else 1)
    train = False

    grad_acc = init_grad_accumulator(params)
    step_counter = 0
    has_guide = model.guide_blosum is not None

    @jax.jit
    def _step(state, batch):
        """
        batch: tuple of arrays from generator:
            batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs
        """
        batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs, batch_refmap = batch
        seq_padding_mask = (batch_seqs == tokens.index(pad_token)).astype(bool)
        #batch_refmap = make_ref_map(batch_alns)
        batch_spanmask = make_span_mask(batch_spans)
        maps, traceback = forward(
            {"params": params},
            batch_seqs,
            seq_padding_mask,
            train
        )
        loss, aln_loss, matchloss, mmatchloss, signalloss = loss_fn(
            maps,
            traceback,
            batch_spans,
            batch_alns,
            batch_spanmask,
            batch_refmap,
            temp,
            abs_scale,
            max_scale,
            with_traceback
        )
        loss = jnp.mean(loss)
        aln_loss = jnp.mean(aln_loss)
        matchloss = jnp.mean(matchloss)
        mmatchloss = jnp.mean(mmatchloss)
        signalloss = jnp.mean(signalloss)
        sim_loss = sim_loss_fn(
            params["tokenizer"]["embedding"]["embedding"],
            model.guide_blosum,
            model.normalized
        ) * guide_scale if has_guide else 0.
        return loss + sim_loss, (aln_loss, matchloss, mmatchloss, signalloss, sim_loss)

    state = State(params, optimizer_state, grad_acc, step_counter)
    # Iterate over batches
    for batch in jax_batch_generator(dataset, batch_size=batch_size, indices=indices, shuffle=shuffle):
        batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs = batch
        batch_refmap = make_ref_map(batch_alns)
        batch = (batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs, batch_refmap)
        loss_value, aux_losses = _step(state, batch)
        if jnp.isnan(loss_value):
            logging("NaN detected in loss!")
        if has_nan(params):
            logging("NaN detected in model parameters!")

        aln_loss, match_loss_val, mmatch_loss, signal_loss_val, sim_loss_val = aux_losses
        total_loss += loss_value
        total_aloss += aln_loss
        total_mloss += match_loss_val
        total_mmloss += mmatch_loss
        total_sigloss += signal_loss_val
        total_simloss += sim_loss_val

        global_step += 1

    metrics = {
        "total": total_loss / num_batch,
        "aln": total_aloss / num_batch,
        "match": total_mloss / num_batch,
        "mmatch": total_mmloss / num_batch,
        "sigloss": total_sigloss / num_batch,
        "simloss": total_simloss / num_batch
    }

    return metrics

def test_loop_pmap(
    params, optimizer_state, dataset, model, forward, loss_fn, grad_acc_iter, global_step, batch_size,
    shuffle=True, logging=print,
    indices=None, max_grad_norm=1.0, temp=0.08, guide_scale=0.001, abs_scale=0.1,
    num_enc_used=1, max_scale=0.5
    ):
    train = False
    devices = jax.local_devices()
    num_devices = len(devices)

    # Helper: replicate state to devices
    state = State(
        params=params,
        opt_state=optimizer_state,
        grad_acc=init_grad_accumulator(params),
        step_counter=1
    )
    #state_repl = jax.device_put_replicated(state, devices)

    # Shard batch across devices
    def shard(batch):
        return jax.tree_util.tree_map(
            lambda x: x.reshape((num_devices, -1) + x.shape[1:]),
            batch
        )

    # pmap axis name for collective ops
    axis_name = 'batch'

    has_guide = model.guide_blosum is not None
    
    @jax.jit
    def _step(state, batch):
        """
        batch: tuple of arrays from generator:
            batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs
        """
        batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs, batch_refmap = batch
        seq_padding_mask = (batch_seqs == tokens.index(pad_token)).astype(bool)
        #batch_refmap = make_ref_map(batch_alns)
        batch_spanmask = make_span_mask(batch_spans)
        maps, traceback = forward(
            {"params": state.params},
            batch_seqs,
            seq_padding_mask,
            batch_lengths,
            train
        )
        loss, aln_loss, matchloss, mmatchloss, signalloss = loss_fn(
            maps,
            traceback,
            batch_spans,
            batch_alns,
            batch_spanmask,
            batch_refmap,
            temp,
            abs_scale,
            max_scale
        )
        loss = jnp.mean(loss)
        aln_loss = jnp.mean(aln_loss)
        matchloss = jnp.mean(matchloss)
        mmatchloss = jnp.mean(mmatchloss)
        signalloss = jnp.mean(signalloss)
        sim_loss = sim_loss_fn(
            params["tokenizer"]["embedding"]["embedding"],
            model.guide_blosum,
            model.normalized
        ) * guide_scale if has_guide else 0.
        return loss + sim_loss, (aln_loss, matchloss, mmatchloss, signalloss, sim_loss)
    step = jax.pmap(_step, axis_name=axis_name, in_axes=(None, (0, 0, 0, 0, 0, 0, 0)), out_axes=(None, None))
    total_loss = 0.0
    total_aloss = 0.0
    total_mloss = 0.0
    total_mmloss = 0.0
    total_sigloss = 0.0
    total_simloss = 0.0
    n = len(indices if indices is not None else dataloader)
    num_batch = n // batch_size + (0 if n % batch_size == 0 else 1)
    # Iterate over batches
    for batch in jax_batch_generator(dataset, batch_size=batch_size, indices=indices, shuffle=shuffle):
        batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs = batch
        batch_refmap = make_ref_map(batch_alns)
        batch_full = (batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs, batch_refmap)
        
        # Shard the batch for devices
        batch_sharded = shard(batch_full)

        # Run parallel step
        loss_value, aux_losses = step(state, batch_sharded)
        if jnp.isnan(loss_value):
            logging("NaN detected in loss!")
        if has_nan(params):
            logging("NaN detected in model parameters!")
        loss_value = jnp.mean(loss_value)
        aux_losses = jax.tree_util.tree_map(lambda x: jnp.mean(x), aux_losses)

        aln_loss, match_loss_val, mmatch_loss, signal_loss_val, sim_loss_val = aux_losses
        total_loss += float(loss_value)
        total_aloss += float(aln_loss)
        total_mloss += float(match_loss_val)
        total_mmloss += float(mmatch_loss)
        total_sigloss += float(signal_loss_val)
        total_simloss += float(sim_loss_val)

        global_step += 1

    metrics = {
        "total": total_loss / num_batch,
        "aln": total_aloss / num_batch,
        "match": total_mloss / num_batch,
        "mmatch": total_mmloss / num_batch,
        "sigloss": total_sigloss / num_batch,
        "simloss": total_simloss / num_batch
    }

    return metrics


def plot_samples(params, dataset, forward, writer, indices, prefix, global_step, max_scale):
    @jax.jit
    def _step(params, batch):
        """
        batch: tuple of arrays from generator:
            batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs
        """
        train = False
        batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs = batch
        seq_padding_mask = (batch_seqs == tokens.index(pad_token)).astype(bool)
        maps, traceback = forward(
            {"params": params},
            batch_seqs,
            seq_padding_mask,
            batch_lengths,
            train
        )
        return maps, batch_refmap, traceback

    def reshape(arr):
        assert(len(arr.shape) == 2)
        return np.array(jnp.expand_dims(arr, 0))

    for batch in jax_batch_generator(dataset, batch_size=len(indices), indices=indices, shuffle=False):
        batch_seqs, batch_lengths, batch_spans, batch_alns, batch_scores, batch_idxs = batch
        batch_refmap = make_ref_map(batch_alns)

        maps, ref_map, traceback = _step(params, batch)
        for i, m, rm, tb in zip(batch_idxs, maps, ref_map, traceback):
            writer.add_image(f"Samples/sample_{prefix}_{i}/map", reshape(m), global_step)
            writer.add_image(f"Samples/sample_{prefix}_{i}/refmap", reshape(rm), global_step)
            writer.add_image(f"Samples/sample_{prefix}_{i}/traceback", reshape(tb), global_step)
            #writer.add_image(f"Samples/sample_{prefix}_{i}/traceback_normalized", reshape((tb-tb.min())/(tb.max()-tb.min()+1e-8)), global_step)

def main(
    train_file: str,
    train_matches: str,
    cached_train: str,
    valid_file: str,
    valid_matches: str,
    cached_valid: str,
    batch_size: int,
    grad_acc_iter: int,
    epochs: int,
    learning_rate: float,
    output_prefix: str,
    save_interval: int,
    log_interval: int,
    checkpoint: str,
    hyperparameter: dict,
    model_parameters: dict
):
    model = JaxEncoder(
        **model_parameters
    )
    key = jax.random.PRNGKey(0)
    dummy_seq = jnp.zeros((2, max_length), dtype=jnp.uint8)  # shape: (2, L)
    dummy_padmask = jnp.zeros((2, max_length), dtype=bool)
    dummy_lengths = jnp.array([max_length, max_length], dtype=jnp.int32)
    variables = model.init(key, dummy_seq, dummy_padmask, dummy_lengths, with_traceback=False, train=False)
    #batch_stats = {} #variables["batch_stats"]
    params = variables["params"]
    """for k, v in traverse_util.flatten_dict(params).items():
        print(k, v.shape)
    """

    logging = logger(base_out + outfile)
    #logging(params)
    # Load the dataset
    dataset = JaxDataset(train_file, train_matches, max_length=max_length, h5file=cached_train, index1=True)
    subset_start, subset_end = 0, len(dataset) - 1
    subset_size = 1200000 #int(len(dataset) * 0.9)
    train_indices = random.sample(range(len(dataset)), subset_size)
    num_train_samples = 20
    train_samples = train_indices[:num_train_samples]

    testset = JaxDataset(valid_file, valid_matches, max_length=max_length, h5file=cached_valid, index1=True)
    testsubset_start, testsubset_end = 0, len(testset) - 1
    test_subset_size = 120000
    test_indices = random.sample(range(len(testset)), test_subset_size)
    num_test_samples = 20
    test_samples = test_indices[:num_test_samples]

    mask = decay_mask(params)  # Boolean mask PyTree matching params

    clip_norm = 1.0
    train_steps_per_epoch = subset_size // batch_size + 1
    total_steps = epochs * train_steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps
    )

    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0.0,
        transition_steps=total_steps - warmup_steps
    )

    schedule_fn = optax.join_schedules([warmup_fn, decay_fn], [warmup_steps])

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),  # gradient clipping
        optax.adamw(learning_rate=schedule_fn, weight_decay=0.01, mask=mask)
    )
    optimizer_state = optimizer.init(params)

    if checkpoint:
        params, optimizer_state, trained_epoch, global_step = load_checkpoint(checkpoint)
    else:
        trained_epoch, global_step = 0, 0

    num_batch = subset_size // batch_size + (1 if subset_size % batch_size != 0 else 0)
    logging(f"Training on {num_batch} batches, each batch {batch_size} samples, totaling {subset_size} samples")
    logging(model_parameters)
    logging("Epoch\tLoss\tUniqueAln\tSigAln\tTime")
    log_t = 0
    log_dir = None
    writer = None
    temp = 0.07
    guide_scale = 0.1
    abs_scale = 0.3
    max_scale = 0.2

    exp_name = f"experiment_ssw_{batch_size}_{learning_rate}_{get_model_info_str(model_parameters)}_{datetime.datetime.now().strftime('%m%d-%H%M')}"
    exp_path = "runs/" + exp_name

    batched_forward_train = jax.jit(jax.vmap(
        lambda params, seq, seq_padmask, lengths, train: model.apply(params, seq, seq_padmask, lengths, train=True),
        in_axes=(None, 0, 0, 0, None)
    ))
    batched_forward_test = jax.jit(jax.vmap(
        lambda params, seq, seq_padmask, lengths, train: model.apply(params, seq, seq_padmask, lengths, train=False),
        in_axes=(None, 0, 0, 0, None)
    ))
    batched_loss_fn = jax.jit(jax.vmap(
        lambda mats, traceback, spanref, seqmatchmask, seqspanmask, pairmatchmask,
            temp, abs_scale, max_scale: loss_fn(mats, traceback, spanref, seqmatchmask, seqspanmask, pairmatchmask,
            temp, abs_scale, max_scale),
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None)
    ))
    # Training loop
    for epoch in range(trained_epoch, trained_epoch + epochs):
        t = time.time()
        params, optimizer_state, metrics, global_step = train_loop_pmap(
            params,
            optimizer,
            optimizer_state,
            dataset,
            model,
            batched_forward_train,
            batched_loss_fn,
            grad_acc_iter=grad_acc_iter,
            global_step=global_step,
            batch_size=batch_size,
            shuffle=True,
            indices=train_indices,
            max_grad_norm=clip_norm,
            temp=temp,
            guide_scale=guide_scale,
            abs_scale=abs_scale,
            max_scale=max_scale,
            logging=logging
        )

        total_loss = metrics["total"]
        total_aloss = metrics["aln"]
        total_mloss = metrics["match"]
        total_mmloss = metrics["mmatch"]
        total_sigloss = metrics["sigloss"]
        total_simloss = metrics["simloss"]

        logging(f"{epoch + 1}/{trained_epoch + epochs}\t{total_loss:.4f}\t{total_aloss:.4f}\t{total_mloss:.4f}\t{(time.time()-t)/3600:.2f}")

        if (epoch+1) % save_interval == 0:
            # Save the model weights
            output_state = output_prefix + "_{}".format(epoch+1) + '.pth'
            save_checkpoint(params, optimizer_state, epoch, global_step, output_state)

        if (epoch+1) % log_interval == 0:
            if not log_dir:
                log_dir = base_out + exp_path
                writer = SummaryWriter(log_dir)
            writer.add_scalar("Loss/train_total", float(total_loss), global_step)
            writer.add_scalar("Loss/train_aln", float(total_aloss), global_step)
            writer.add_scalar("Loss/train_match", float(total_mloss), global_step)
            writer.add_scalar("Loss/train_mmatch", float(total_mmloss), global_step)
            writer.add_scalar("Loss/train_sigloss", float(total_sigloss), global_step)
            writer.add_scalar("Loss/train_simloss", float(total_simloss), global_step)
            #writer.add_scalar("Hyperparameter/LR", scheduler.get_last_lr()[0], global_step)
            plot_samples(
                params,
                dataset,
                batched_forward_test,
                writer,
                train_samples,
                'train',
                global_step,
                max_scale
            )
            if test_subset_size <= 0:
                continue
            tt = time.time()
            metrics = test_loop_pmap(
                params,
                optimizer_state,
                testset,
                model,
                batched_forward_test,
                batched_loss_fn,
                grad_acc_iter=grad_acc_iter,
                global_step=global_step,
                batch_size=batch_size,
                shuffle=False,
                indices=test_indices,
                max_grad_norm=clip_norm,
                temp=temp,
                guide_scale=guide_scale,
                abs_scale=abs_scale,
                max_scale=max_scale,
                logging=logging
            )
            total_tloss = metrics["total"]
            total_taloss = metrics["aln"]
            total_tmloss = metrics["match"]
            total_tmmloss = metrics["mmatch"]
            total_tsigloss = metrics["sigloss"]
            total_tsimloss = metrics["simloss"]
            writer.add_scalar("Loss/test_total", float(total_tloss), global_step)
            writer.add_scalar("Loss/test_aln", float(total_taloss), global_step)
            writer.add_scalar("Loss/test_match", float(total_tmloss), global_step)
            writer.add_scalar("Loss/test_mmatch", float(total_tmloss), global_step)
            writer.add_scalar("Loss/test_sigloss", float(total_tsigloss), global_step)
            writer.add_scalar("Loss/test_simloss", float(total_tsimloss), global_step)

            plot_samples(
                params,
                testset,
                batched_forward_test,
                writer,
                test_samples,
                'test',
                global_step,
                max_scale
            )
            logging(f"Test\t{total_tloss:.4f}\t{total_taloss:.4f}\t{total_tmloss:.4f}\t{(time.time() - tt)/3600:.2f}")
        
        if total_mloss < 1.3:
            logging("with_traceback")
            max_scale = 0.05

if __name__ == "__main__":
    '''train_file = base_data + 'dataproc_direct.fasta'
    train_matches = base_data + 'dataproc_direct_matches.h5'
    cached_train = base_data + 'cached_dataproc_direct.h5'
    valid_file = base_data + 'dataproc_direct.fasta'
    valid_matches = base_data + 'dataproc_direct_matches.h5'
    cached_valid = base_data + "cached_dataproc_direct.h5"'''
    train_file = base_data + 'uniref50_3m_3m.fasta'
    train_matches = base_data + 'uniref50_3m_3m_matches.h5'
    cached_train = base_data + "cached_3m_3m.h5"
    valid_file = base_data + 'uniref50_1m_1m_01.fasta'
    valid_matches = base_data + 'uniref50_1m_1m_01_matches.h5'
    cached_valid = base_data + "cached_1m_1m_01.h5"
    batch_size = 100
    grad_acc_iter = 5
    epochs = 3
    learning_rate = 0.0005
    output_state = base_ckpts + 'ckpts_ssw_ce_1.2m_48+144+240+384+480_7+9+11+13+15_1+2+2+3+4_480'
    save_interval = 1
    log_interval = 1
    checkpoint = None
    '''
    print(jax.default_backend())      # 'cpu' or 'gpu'
    print(jax.devices())              # list of available devices
    print(jax.devices()[0].platform)  # 'cpu' / 'gpu'
    '''
    #model parameters
    model_parameters = {
        'dims': [48, 144, 240, 384, 480],
        'conv_kernels':[7, 9, 11, 13, 15],
        'dilations': [1, 2, 2, 3, 4],
        'init_token': None, #get_mdsdecmp()
        'guide_blosum': jnp.array(blosum62_gttl, dtype=jnp.float32),
        'normalized': True,
        'outdim': 480,
        'ssw_matrix': jnp.array(blosum62_gttl, dtype=jnp.float32),
        'ssw_unroll': 2,
        'gap_open': -11.,
        'gap_ext': -1.,
        'ssw_NINF': -1e8,
        'ssw_eps': 0,
        'ssw_temp': 0.8,
        'ssw_restrict_turns': True,
        'ssw_penalize_turns': True
    }

    main(
        train_file=train_file,
        train_matches=train_matches,
        cached_train=cached_train,
        valid_file=valid_file,
        valid_matches=valid_matches,
        cached_valid=cached_valid,
        batch_size=batch_size,
        grad_acc_iter=grad_acc_iter,
        epochs=epochs,
        learning_rate=learning_rate,
        output_prefix=output_state,
        save_interval=save_interval,
        log_interval=log_interval,
        checkpoint=checkpoint,
        hyperparameter={},
        model_parameters=model_parameters
    )
