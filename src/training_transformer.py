import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import argparse as ap
from dataset import ColBertDatasetAln_direct
from transformer_encoder import ColBERT_transformer
from constants import max_length, input_dim, latent_dim, hidden_dim, TOKENS, attention_mask_window_size, PAD_TOKEN, blosum62_gttl
from utils import tokens, pad_token
from mds import get_mdsdecmp
import numpy as np
import cv2
#769228, 585305, 699545, 388597
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
base_data = '/scratch/stud2018/ata/ma_data/'
base_ckpts = '/scratch/stud2018/ata/'
base_out = '/scratch/stud2018/ata/'
outfile = "out2.csv"

def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1
        #return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, path="checkpoint.pth"):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, scheduler, scaler, path="checkpoint.pth", device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]
    print(f"Checkpoint loaded from {path}, resuming from epoch {epoch}, step {global_step}")
    return epoch, global_step

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def get_decay_factor(start,end,duration):
    return (end/start) ** (1/duration)

def get_attention_mask(window_size=max_length, seq_len=max_length):
    mask = torch.full((seq_len, seq_len), True, dtype=bool)

    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = False

    return mask

def make_ref_map(alnref):
    def make_map(qaln, daln):
        B, n = qaln.shape
        m = torch.zeros((B,n,n),dtype=torch.bool)
        qidxs = torch.argwhere(qaln)
        didxs = torch.argwhere(daln)
        assert(qidxs.size(0) == didxs.size(0))
        for (bq, qidx), (bd, didx) in zip(qidxs, didxs):
            assert(bq == bd)
            m[bq, qidx, didx] = True
        return m
    return make_map(alnref[:,0,:],alnref[:,1,:]).to(device)

def make_ref_map2(alnref):
    def make_map(qaln, daln):
        B, n = qaln.shape
        m = torch.zeros((B, n, n), dtype=torch.bool, device=qaln.device)

        # Get indices where qaln and daln are True
        qidxs = torch.nonzero(qaln, as_tuple=False)  # (K, 2) [batch, index]
        didxs = torch.nonzero(daln, as_tuple=False)  # (K, 2)

        assert qidxs.size(0) == didxs.size(0)
        assert torch.all(qidxs[:, 0] == didxs[:, 0])  # batch indices align

        # Unpack indices for advanced assignment
        b = qidxs[:, 0]
        qi = qidxs[:, 1]
        di = didxs[:, 1]

        # Vectorized scatter assignment
        m[b, qi, di] = True
        return m
    return make_map(alnref[:,0,:],alnref[:,1,:]).to(device)

def reverse_seq(seqs, seqlen):
    B, S, L = seqs.shape
    pad_value = TOKENS.index(PAD_TOKEN)
    assert(S in [3,4])
    if S == 3:
        seqlen_tmp = seqlen.view((B * 3)).unsqueeze(1).int()
    else:
        seqlen_tmp = torch.empty((B, 4), device=seqlen.device)
        seqlen_tmp[:,0] = seqlen[:,0]
        seqlen_tmp[:,1] = seqlen[:,1]
        seqlen_tmp[:,2] = seqlen[:,0]
        seqlen_tmp[:,3] = seqlen[:,2]
        seqlen_tmp = seqlen_tmp.view((B * 4)).unsqueeze(1).int()

    seqs = seqs.view((B * S, L))
    arange = torch.arange(L, device=seqlen_tmp.device).expand(B * S, L).int()
    mask = arange < seqlen_tmp

    # Compute reversed indices per batch
    rev_idx = (seqlen_tmp - 1 - arange)
    rev = torch.gather(seqs, 1, rev_idx.to(torch.int64))  # reverse each row
    seqs = torch.where(mask, rev, torch.full_like(seqs, pad_value))
    return seqs.view((B, S, L))

def make_soft_heatmap(mask, sigma=1.0, eps=1e-6):
    mask = mask.unsqueeze(1)
    N, C, H, W = mask.shape
    mask_np = mask.cpu().numpy().astype(np.float32)  # (N,1,H,W)
    heatmaps = []

    for i in range(N):
        img = mask_np[i,0]  # (H,W)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=sigma, sigmaY=sigma)
        # Normalize to [0,1]
        if blurred.max() > 0:
            blurred = blurred / blurred.max()
        heatmaps.append(blurred)  # add channel dim

    heatmaps = np.stack(heatmaps, axis=0)  # (N,1,H,W)
    heatmaps = torch.from_numpy(heatmaps).view(N,-1).to(mask.device)
    heatmaps = heatmaps / (heatmaps.sum(dim=1, keepdim=True) + eps)
    heatmaps = heatmaps.view(N, H, W)
    return heatmaps

def gaussian_blur1d(targets, sigma=2.0):
    """
    targets: (N, L), each row a hard probability distribution
    """
    B, C, L = targets.shape
    assert(C == 4 and L == max_length)
    targets = targets.float().view((B*C,L))
    ksize = int(6 * sigma + 1)  # cover +/- 3σ
    if ksize % 2 == 0:
        ksize += 1
    x = torch.arange(ksize, device=targets.device) - ksize // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()   # normalize
    kernel = kernel.view(1, 1, -1)   # (out_ch, in_ch, k)

    # conv with reflect padding, apply same kernel to each batch
    targets_ = targets.unsqueeze(1)   # (N,1,L)
    smoothed = F.conv1d(F.pad(targets_, (ksize//2, ksize//2), mode="reflect"),
                        kernel)
    smoothed = smoothed.squeeze(1)
    targets = targets.view((B,C,L))
    # renormalize to sum=1
    return smoothed / (smoothed.sum(dim=1, keepdim=True) + 1e-8)

def span_to_coord(span, seqlen):
    B = span.size(0)
    coords = torch.empty((B,2,4), dtype=span.dtype, device=span.device)
    coords[:,0,0] = span[:,0,0]
    coords[:,0,1] = span[:,1,0]
    coords[:,0,2] = seqlen[:,0] - span[:,0,1] - 1
    coords[:,0,3] = seqlen[:,1] - span[:,1,1] - 1
    coords[:,1,0] = span[:,2,0]
    coords[:,1,1] = span[:,3,0]
    coords[:,1,2] = seqlen[:,0] - span[:,2,1] - 1
    coords[:,1,3] = seqlen[:,2] - span[:,3,1] - 1
    return coords

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        #print(param)
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        #print(buffer)
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def make_span_mask(span, maxlen=max_length):
    B, C, D = span.shape
    assert(C == 2 and D == 2)
    starts = span[:,:, 0].unsqueeze(2)  # [N, 1]
    ends   = span[:,:, 1].unsqueeze(2)  # [N, 1]
    idx = torch.arange(maxlen,device=span.device).unsqueeze(0).unsqueeze(0)  # [1, length]
    mask = (idx >= starts) & (idx <= ends)
    #spanmask = torch.zeros((B,C,maxlen),dtype=bool,device=span.device)
    #for i in range(C):
    #    for j in range(B):
    #        spanmask[j,i,span[j,i,0]:span[j,i,1]+1] = 1
    return mask

def make_sim_distribution(S, tau=1, per_row_normalize=False):
    if per_row_normalize:
        mu = S.mean(dim=1, keepdim=True)
        sigma = S.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
        S = (S - mu) / sigma
    S = S - S.max(dim=1, keepdim=True).values   # subtract row max for stability
    #print(S)
    logits = S / tau
    #print(logits)
    P = F.softmax(logits, dim=1)
    #print(P)                # (24,24)
    return P

def get_model_info_str(model_params, l=['!'], depth=0):
    l = ["-","+","=",'*',"#"]
    def make_str(attr, depth=0):
        if depth >= len(l):
            raise RuntimeError("Out of depth")
        return l[depth].join([str(a) if not isinstance(a,list) else make_str(a, depth+1) for a in attr])
    return "_".join([
        make_str(model_params["dims"]),
        make_str(model_params["conv_kernels"]),
        make_str(model_params["dilations"]),
        make_str(model_params["outdims"]),
        str(model_params['num_layers']),
        str(model_params['num_heads']),
        str(model_params['mlp_ratio'])
    ])

def main(
        train_file: str,
        train_matches: str,
        cached_train: str,
        valid_file: str,
        valid_matches: str,
        cached_valid: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        output_prefix: str,
        save_interval: int,
        log_interval: int,
        checkpoint: str,
        hyperparameter: dict,
        model_parameters: dict
):
    model = ColBERT_transformer(
        **model_parameters
    )
    print('Model size: {:.3f}MB'.format(get_model_size(model)))
    with open(base_out + outfile, "a") as file:
        file.write('Model size: {:.3f}MB\n'.format(get_model_size(model)))
    # Load the dataset
    dataset = ColBertDatasetAln_direct(train_file, train_matches, max_length=max_length, h5file=cached_train, padding=0, index1=True)
    subset_start, subset_end = 0, len(dataset) - 1
    subset_size = 20000000 #int(len(dataset) * 0.9)
    train_indices = random.sample(range(len(dataset)), subset_size)
    train_subset = Subset(dataset, train_indices)
    dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=False)
    num_train_samples = 20
    train_samples = train_indices[:num_train_samples]

    testset = ColBertDatasetAln_direct(valid_file, valid_matches, max_length=max_length, h5file=cached_valid, padding=0, index1=True)
    testsubset_start, testsubset_end = 0, len(dataset) - 1
    test_subset_size = 2000000 #len(dataset) - subset_size
    test_indices = random.sample(range(len(testset)), test_subset_size)
    test_subset = Subset(testset, test_indices)
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=True, pin_memory=False)
    num_test_samples = 20
    test_samples = test_indices[:num_test_samples]

    model = torch.compile(model, mode="default", fullgraph=False)
    model.apply(init_weights)
    model = model.to(device)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen parameters
        if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = optim.AdamW([
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=learning_rate, weight_decay=0.01)

    #if torch.cuda.device_count() > 1:
    #    model = DataParallel(model)

    scaler = GradScaler('cuda')  # For FP16
    clip_norm = 1.0

    train_steps_per_epoch = subset_size // batch_size + 1
    total_steps = epochs * train_steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    if checkpoint:
        trained_epoch, global_step = load_checkpoint(model, optimizer, scheduler, scaler, checkpoint, device=device)
    else:
        trained_epoch, global_step = 0, 0

    num_batch = subset_size // batch_size + (1 if subset_size % batch_size != 0 else 0)
    print(f"Training on {num_batch} batches, each batch {batch_size} samples, totaling {subset_size} samples")
    with open(base_out + outfile, "a") as file:
        file.write(f"Training on {num_batch} batches, each batch {batch_size} samples, totaling {subset_size} samples\n")
    print(model_parameters)
    with open(base_out + outfile, "a") as file:
        file.write(f"{model_parameters}\n")
    print("Epoch\tLoss\tUniqueAln\tSigAln\tTime")
    with open(base_out + outfile, "a") as file:
        file.write("Epoch\tLoss\tUniqueAln\tSigAln\tTime\n")
    log_t = 0
    log_dir = None
    writer = None
    temp = 0.07
    guide_scale = 0.01
    abs_scale = 0.1
    num_enc_used = 1
    threshold = 0.8
    max_scale = 2
    end_mscale = 3
    max_scale_f = get_decay_factor(max_scale, end_mscale, 3)

    attn_mask = get_attention_mask(20).to(device)

    exp_name = f"trans_{datetime.datetime.now().strftime('%m%d-%H%M')}_{batch_size}_{learning_rate}_{temp}_{get_model_info_str(model_parameters)}"
    exp_path = "runs/" + exp_name
    # Training loop
    for epoch in range(trained_epoch, trained_epoch + epochs):
        model.train()
        total_loss = 0
        total_aloss = 0
        total_mloss = 0
        total_mmloss = 0
        total_sigloss = 0
        total_simloss = 0
        t = time.time()
        rt = 0
        for batch_idx, (seq, seqlen, spanref, alnref, _, idx) in enumerate(dataloader):
            seq = seq.to(device)
            seqlen = seqlen.to(device)
            spanref = spanref.to(device)
            alnref = alnref.to(device)
            spanmask = make_span_mask(spanref)
            ref_map = make_ref_map2(alnref)
            seq_padding_mask = (seq == tokens.index(pad_token)).bool()
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda',dtype=torch.float16):  # Enable FP16
                tt = time.time()
                maps = model(seq, attn_mask, seq_padding_mask, num_enc_used)
                if (maps > 1.1).any() or (maps < -1.1).any():
                    print("Detected anomalies in sim matrix")
                    maps = torch.clamp(maps, min=-1, max=1)
                loss, aln_loss, matchloss, mmatchloss, signalloss, simloss = model.loss_fn(
                    maps,
                    spanref,
                    alnref,
                    spanmask,
                    ref_map.unsqueeze(1),
                    temp=temp,
                    guide_scale=guide_scale,
                    abs_scale=abs_scale,
                    num_enc_used=num_enc_used,
                    threshold=threshold,
                    max_scale=max_scale
                )
            if (epoch+1) % log_interval == 0:
                if not log_dir:
                    log_dir = base_out + exp_path
                    writer = SummaryWriter(log_dir)
                for i, m, rm in zip(idx, maps, ref_map):
                    if i in train_samples:
                        for j in range(m.shape[0]):
                            writer.add_image(f"Samples/sample_train_{i}/map_{j}", m[j,:,:].unsqueeze(0), global_step)
                        writer.add_image(f"Samples/sample_train_{i}/totalmap", temp*torch.logsumexp(m/temp, dim=0).unsqueeze(0), global_step)
                        writer.add_image(f"Samples/sample_train_{i}/refmap", rm.unsqueeze(0), global_step)
                        #writer.add_image(f"Samples/sample_train_{i}/traceback", ((tb-tb.min())/(tb.max()-tb.min())).unsqueeze(0), global_step)

            if torch.isnan(loss).any():
                print("NaN detected in logits!")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            rt += time.time() - tt

            total_loss += loss.item()
            total_aloss += aln_loss.item()
            total_mloss += matchloss.item()
            total_mmloss += mmatchloss.item()
            total_sigloss += signalloss.item()
            total_simloss += simloss.item()

            global_step += 1

            if subset_size >= 1000000 and global_step % ((num_batch // 10) if (num_batch // 10) > 0 else 1) == 0:
                if not log_dir:
                    log_dir = base_out + exp_path
                    writer = SummaryWriter(log_dir)
                writer.add_scalar("Loss/train_total", total_loss / (batch_idx+1),  global_step)
                writer.add_scalar("Loss/train_aln", total_aloss / (batch_idx+1), global_step)
                writer.add_scalar("Loss/train_match", total_mloss / (batch_idx+1), global_step)
                writer.add_scalar("Loss/train_mmatch", total_mmloss / (batch_idx+1), global_step)
                writer.add_scalar("Loss/train_sigloss", total_sigloss / (batch_idx+1), global_step)
                writer.add_scalar("Loss/train_simloss", total_simloss / (batch_idx+1) , global_step)
                writer.add_scalar("Hyperparameter/LR", scheduler.get_last_lr()[0], global_step)

            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN in weights: {name}")

        print(f"{epoch + 1}/{trained_epoch + epochs}\t{total_loss / len(dataloader):.6f}\t{total_aloss/len(dataloader):.6f}\t{total_mloss/len(dataloader):.6f}\t{(time.time()-t)/3600:.2f}")
        with open(base_out + outfile, "a") as file:
            file.write(f"{epoch + 1}/{trained_epoch + epochs}\t{total_loss / len(dataloader):.6f}\t{total_aloss/len(dataloader):.6f}\t{total_mloss/len(dataloader):.6f}\t{(time.time()-t)/3600:.2f}\n")

        if (epoch+1) % save_interval == 0:
            # Save the model weights
            output_state = output_prefix + "_{}".format(epoch+1) + '.pth'
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, output_state)

        if (epoch+1) % log_interval == 0:
            if not log_dir:
                log_dir = base_out + exp_path
                writer = SummaryWriter(log_dir)
            writer.add_scalar("Loss/train_total", total_loss / len(dataloader), global_step)
            writer.add_scalar("Loss/train_aln", total_aloss / len(dataloader), global_step)
            writer.add_scalar("Loss/train_match", total_mloss / len(dataloader), global_step)
            writer.add_scalar("Loss/train_sigloss", total_sigloss / len(dataloader), global_step)
            writer.add_scalar("Loss/train_simloss", total_simloss / len(dataloader), global_step)
            writer.add_scalar("Hyperparameter/LR", scheduler.get_last_lr()[0], global_step)
            log_t = time.time()
            if test_subset_size <= 0:
                continue
            model.eval()
            total_tloss = 0
            total_tmloss = 0
            total_tmmloss = 0
            total_tsigloss = 0
            total_tsimloss = 0
            total_taloss = 0
            with torch.no_grad():
                for seq, seqlen, spanref, alnref, target_score, idx in testloader:
                    seq = seq.to(device)
                    seqlen = seqlen.to(device)
                    spanref = spanref.to(device)
                    target_score = target_score.to(device)
                    spanmask = make_span_mask(spanref)
                    alnref = alnref.to(device)
                    ref_map = make_ref_map2(alnref)
                    seq_padding_mask = (seq == tokens.index(pad_token)).bool()
                    with autocast(device_type='cuda',dtype=torch.float16):  # Enable FP16
                        maps = model(seq, attn_mask, seq_padding_mask, num_enc_used)
                        #assert (maps <= 1.1).all() and (maps >= -1.1).all(), "Similarity matrix out of range!"
                        if (maps > 1.1).any() or (maps < -1.1).any():
                            print("Detect anomalies in sim matrix")
                            maps = torch.clamp(maps, min=-1, max=1)
                        tloss, taln_loss, tmloss, tmmloss, tsigloss, tsimloss = model.loss_fn(
                            maps,
                            spanref,
                            alnref,
                            spanmask,
                            ref_map.unsqueeze(1),
                            temp=temp,
                            guide_scale=guide_scale,
                            abs_scale=abs_scale,
                            num_enc_used=num_enc_used,
                            threshold=threshold,
                            max_scale=max_scale
                        )
                    for i, m, rm in zip(idx, maps, ref_map):
                        if i in test_samples:
                            for j in range(m.shape[0]):
                                writer.add_image(f"Samples/sample_test_{i}/map_{j}", m[j,:,:].unsqueeze(0), global_step)
                            writer.add_image(f"Samples/sample_test_{i}/totalmap", temp*torch.logsumexp(m/temp, dim=0).unsqueeze(0), global_step)
                            writer.add_image(f"Samples/sample_test_{i}/refmap", rm.unsqueeze(0), global_step)
                            #writer.add_image(f"Samples/sample_test_{i}/traceback", ((tb-tb.min())/(tb.max()-tb.min())).unsqueeze(0).unsqueeze(0), global_step)
                    total_tloss += tloss.item()
                    total_taloss += taln_loss.item()
                    total_tmloss += tmloss.item()
                    total_tmmloss += tmmloss.item()
                    total_tsigloss += tsigloss.item()
                    total_tsimloss += tsimloss.item()
                writer.add_scalar("Loss/test_total", total_tloss / len(testloader), global_step)
                writer.add_scalar("Loss/test_aln", total_taloss / len(testloader), global_step)
                writer.add_scalar("Loss/test_match", total_tmloss / len(testloader), global_step)
                writer.add_scalar("Loss/test_mmatch", total_tmmloss / len(testloader), global_step)
                writer.add_scalar("Loss/test_sigloss", total_tsigloss / len(testloader), global_step)
                writer.add_scalar("Loss/test_simloss", total_tsimloss / len(testloader), global_step)
                print(f"Test\t{total_tloss / len(testloader):.6f}\t{total_taloss/len(testloader):.6f}\t{total_tmloss/len(testloader):.6f}\t0")
                with open(base_out + outfile, "a") as file:
                    file.write(f"Test\t{total_tloss / len(testloader):.6f}\t{total_taloss/len(testloader):.6f}\t{total_tmloss/len(testloader):.6f}\t0\n")

        if num_enc_used < model.num_enc and total_mloss / len(dataloader) < 1.:
            num_enc_used += 1
        max_scale *= max_scale_f

if __name__ == "__main__":
    '''train_file = base_data + 'dataproc_direct.fasta'
    train_matches = base_data + 'dataproc_direct_matches.h5'
    cached_train = base_data + 'cached_dataproc_direct.h5'
    valid_file = base_data + 'dataproc_direct.fasta'
    valid_matches = base_data + 'dataproc_direct_matches.h5'
    cached_valid = base_data + "cached_dataproc_direct.h5"'''
    train_file = base_data + 'uniref50_3m_3m.fasta'
    train_matches = base_data + 'uniref50_3m_3m_matches.h5'
    cached_train = base_data + 'cached_3m_3m.h5'
    valid_file = base_data + 'uniref50_1m_1m_01.fasta'
    valid_matches = base_data + 'uniref50_1m_1m_01_matches.h5'
    cached_valid = base_data + "cached_1m_1m_01.h5"
    batch_size = 320
    epochs = 2
    learning_rate = 0.00005
    output_state = base_ckpts + 'ckpts_trans_pyramid_ptwise_noskip_20m_0.07_128+256_3+3_1+1_3+6_6_8_2_256'
    save_interval = 1
    log_interval = 1
    checkpoint = None

    #model parameters
    model_parameters = {
        'dims': [[128, 256]],
        'conv_kernels':[[3, 3]],
        'dilations': [[1, 1]],
        'enc_type': 0,
        'init_token': get_mdsdecmp(),
        'guide_blosum': torch.tensor(blosum62_gttl, dtype=torch.float32),
        'normalized': True,
        'outdims': [256],
        'skip': False,
        'ptwise': True,
        'extracts': [[3,6]],
        'num_layers': 6,
        'num_heads': 8,
        'mlp_ratio': 2
    }

    main(
        train_file=train_file,
        train_matches=train_matches,
        cached_train=cached_train,
        valid_file=valid_file,
        valid_matches=valid_matches,
        cached_valid=cached_valid,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        output_prefix=output_state,
        save_interval=save_interval,
        log_interval=log_interval,
        checkpoint=checkpoint,
        hyperparameter={},
        model_parameters=model_parameters
    )
