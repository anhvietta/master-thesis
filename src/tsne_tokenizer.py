from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pencoder import ColBERT_direct
from constants import AMINO_ACIDS_GTTL, blosum62_gttl, blosum62_kz, kz_alphabet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

def load_checkpoint(model, optimizer, scheduler, scaler, path="checkpoint.pth", device="cpu", only_model=False):
    checkpoint = torch.load(path, map_location=torch.device(device))
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state"].items() if 'colbert.encoder.pos_encoder' not in k and 'colbert.encoder.encoder' not in k and 'locality_coef' not in k and 'del_per_aa' not in k}
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
    print(f"Checkpoint loaded from {path}, resuming from epoch {epoch}, step {global_step}")
    return epoch, global_step

def tsne(model_path, model_params, out_path):
    model = ColBERT_direct(
        **model_params
    )
    epoch, global_step = load_checkpoint(model, None, None, None, model_path, only_model=True)
    embedding = model.tokenizer.embedding[0].weight.detach().numpy()
    tsne_embedding = PCA(n_components=2).fit_transform(embedding)
    fig = plt.figure()
    ax = fig.add_subplot()
    assert(len(tsne_embedding) == len(AMINO_ACIDS_GTTL))
    for emb, aa in zip(tsne_embedding, AMINO_ACIDS_GTTL):
        ax.scatter(emb[0], emb[1])
        ax.text(emb[0], emb[1], aa, None)

    ax.set_xlabel('TSNE1')
    ax.set_ylabel('TSNE2')
    plt.show()
    plt.savefig(out_path)

    '''tsne_kz = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(blosum62_kz.numpy())
    fig = plt.figure()
    ax = fig.add_subplot()
    assert(len(tsne_kz) == len(kz_alphabet))
    for emb, aa in zip(tsne_kz, kz_alphabet):
        ax.scatter(emb[0], emb[1])
        ax.text(emb[0], emb[1], aa, None)

    ax.set_xlabel('TSNE1')
    ax.set_ylabel('TSNE2')
    #ax.set_zlabel('TSNE3')
    plt.show()
    plt.savefig("/scratch/stud2018/ata/tsne2d_kz.png")'''

if __name__ == "__main__":
    model_path = "/scratch/stud2018/ata/ckpts_96m_32+96+160+256+320+448_7+9+11+13+15+17_1+1+2+2+3+3_372_1.pth"
    model_params = {
        'dims': [[32, 96, 160, 256, 320, 448],[64, 128, 256, 384]],
        'conv_kernels':[[7, 9, 11, 13, 15, 17], [3, 5, 5, 7]],
        'dilations': [[1, 1, 2, 2, 3, 3], [1, 2, 2, 3]],
        'enc_type': 0,
        'init_token': None, #get_mdsdecmp()
        'guide_blosum': torch.tensor(blosum62_gttl, dtype=torch.float32),
        'normalized': True,
        'outdims': [372, 372]
    }
    out_path = "/scratch/stud2018/ata/tsne2d_tokenizer.png"
    tsne(model_path, model_params, out_path)
