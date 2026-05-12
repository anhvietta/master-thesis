"""
Export ProtSearch model from PyTorch to use in LibTorch
"""

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
from dataset import SequenceDataset
from pencoder import ColBERT_direct
from constants import max_length, input_dim, latent_dim, hidden_dim, TOKENS, attention_mask_window_size, PAD_TOKEN, blosum62_gttl
from utils import tokens, pad_token
from mds import get_mdsdecmp
import numpy as np
import cv2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
base_data = '../ma_data/'
base_ckpts = '../ckpts/'
base_out = '../'


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        # print(param)
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        # print(buffer)
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def check_and_fix_undefined_tensors(model: nn.Module, device=None, init_value=1.0):
    """
    Scans all parameters and buffers in the model for undefined (None) tensors.
    Optionally initializes them to a default tensor to make TorchScript saving possible.

    Args:
        model (nn.Module): the PyTorch model to scan
        device (torch.device, optional): device to initialize tensors on
        init_value (float, optional): default value for initializing undefined tensors

    Returns:
        None
    """
    # Check parameters
    for name, param in model.named_parameters():
        if param is None:
            print(f"[PARAM] {name} is None. Initializing to {init_value}.")
            '''new_param = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
            if device:
                new_param = new_param.to(device)
            setattr(model, name, new_param)'''

    # Check buffers
    for name, buf in model.named_buffers():
        if buf is None:
            print(f"[BUFFER] {name} is None. Initializing to {init_value}.")
            '''new_buf = torch.tensor(init_value, dtype=torch.float32)
            if device:
                new_buf = new_buf.to(device)
            model.register_buffer(name, new_buf)'''

    print("Scan complete. All undefined tensors have been fixed (if any).")


def load_checkpoint(model, optimizer, scheduler, scaler, path="checkpoint.pth", device="cuda", only_model=False):
    checkpoint = torch.load(path, map_location=device)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state"].items(
    ) if 'colbert.encoder.pos_encoder' not in k and 'colbert.encoder.encoder' not in k and 'locality_coef' not in k and 'del_per_aa' not in k}
    '''for k in clean_state.keys():
        if 'colbert.encoder.pos_encoder' in k or 'colbert.encoder.encoder' in k:
            del clean_state[k]'''
    model.load_state_dict(clean_state)
    check_and_fix_undefined_tensors(model)
    if not only_model:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
    epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]
    print(
        f"Checkpoint loaded from {path}, resuming from epoch {epoch}, step {global_step}")
    return epoch, global_step


def get_model(
        model_ckpts: str,
        model_params: dict
):
    model = ColBERT_direct(
        **model_params
    )
    print('Model size: {:.3f}MB'.format(get_model_size(model)))
    # model = torch.compile(model, mode="default", fullgraph=True)
    trained_epoch, global_step = load_checkpoint(
        model, None, None, None, model_ckpts, device=device, only_model=True)
    model = model.to(device)
    model.eval()
    prefix = '.'.join(model_ckpts.split('.')[:-1])
    scripted = torch.jit.script(model)
    scripted.save(prefix + ".pt")
    return model


if __name__ == "__main__":
    model_ckpts = base_ckpts + \
        'ckpts.pth'
    model_params = {
        'dims': [[32, 96, 160, 256, 320, 384, 480]],
        'conv_kernels': [[3, 5, 7, 9, 11, 15, 17]],
        'dilations': [[1, 2, 3, 4, 5, 6, 7]],
        'enc_type': 0,
        'init_token': None,  # get_mdsdecmp(),
        'guide_blosum': torch.tensor(blosum62_gttl, dtype=torch.float32),
        'knn': 5,
        'sigma': None,
        'normalized': True,
        'outdims': [320],
        'skip': False,
        'ptwise': True,
        'extracts': [[0, 2, 4, 6]],
    }
    model = get_model(model_ckpts=model_ckpts, model_params=model_params)
