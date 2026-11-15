pass

import copy
import sys
import types
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MyDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

_fake_util = types.ModuleType('util')
_fake_util.MyDataSet = MyDataSet
sys.modules['util'] = _fake_util

dic = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'C': 3, 'N': 4, 'O': 5,
       'F': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11,
       '=': 12, '#': 13, '(': 14, ')': 15}
vocab_size = len(dic)
BOS_IDX = dic['BOS']
EOS_IDX = dic['EOS']
PAD_IDX = dic['PAD']

seed = 42
torch.manual_seed(seed)

def padding_mask(sentence, pad_idx=0):
    return sentence == pad_idx

def subsequent_mask(size):
    attn_shape = (size, size)
    subsequent_mask_ = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask_) == 1

def collate_fn(batch_data):
    en_input   = []
    de_input   = []
    label_idx  = []
    batch_length = []

    for j, k in batch_data:
        en_input.append(j)
        de_input.append(k[:-1])
        label_idx.append(k[1:])
        batch_length.append(len(k[1:]))

    input_x = torch.stack([
        j if isinstance(j, torch.Tensor) else torch.tensor(j)
        for j in en_input
        ])
    batch_max = max(batch_length)

    de_in = []
    for smile_id in de_input:
        smile_id = smile_id + [0] * (batch_max - len(smile_id))
        de_in.append(smile_id)
    de_in = torch.tensor(de_in)

    pad_mask = padding_mask(de_in, pad_idx=0)
    sub_mask = subsequent_mask(batch_max)

    labels = []
    for label in label_idx:
        label = label + [0] * (batch_max - len(label))
        labels.append(label)
    labels = torch.tensor(labels)

    return input_x, de_in, pad_mask, sub_mask, labels

class Loss(nn.Module):
    def __init__(self, model):
        super(Loss, self).__init__()
        self.model     = model
        self.type_loss = nn.CrossEntropyLoss()

    def forward(self, en, de_1, tgt_mask, tgt_key_padding_mask, label1):
        pre_type  = self.model(en, de_1, tgt_mask, tgt_key_padding_mask)
        type_loss = self.type_loss(
            pre_type.view(-1, pre_type.size(-1)), label1.view(-1))
        return type_loss, pre_type.detach()

class TypeAccuracy(nn.Module):
    def __init__(self, eos_idx=2):
        super(TypeAccuracy, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.eos_idx = eos_idx

    def forward(self, pre_types, labels_type):
        pre_types = self.softmax(pre_types)
        correct_smile = []
        error_smile   = []
        accuracy = 0
        for i in range(len(pre_types)):
            a = torch.argmax(pre_types[i], dim=-1)
            b = labels_type[i]
            if self.eos_idx in b:
                eos_pos = (b == self.eos_idx).nonzero(as_tuple=True)[0][0].item()
                a = a[:eos_pos + 1]
                b = b[:eos_pos + 1]
            if a.equal(b):
                accuracy += 1
                correct_smile.append([a.tolist(), b.tolist()])
            else:
                error_smile.append([a.tolist(), b.tolist()])
        return accuracy, correct_smile, error_smile

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)

class Linear(nn.Linear):
    def __init__(self, d_in: int, d_out: int,
                 bias: bool = True, init: str = "default"):
        super(Linear, self).__init__(d_in, d_out, bias=bias)
        self.use_bias = bias
        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)
        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std   = (scale ** 0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                self.bias.fill_(1.0)

class SEBlock1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        hidden = max(1, channels // reduction)
        self.excite   = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        avg   = self.avg_pool(x).squeeze(-1)
        mx    = self.max_pool(x).squeeze(-1)
        scale = torch.sigmoid(
            self.excite(avg) + self.excite(mx))
        return x * scale.unsqueeze(-1)

class InterleavedTransformerBlock(nn.Module):
    def __init__(self, channels, n_heads, dim_feedforward,
                 dropout, max_len):
        super().__init__()
        self.pe         = PositionalEncoding(channels, dropout, max_len)
        self.norm1      = nn.LayerNorm(channels)
        self.self_attn  = nn.MultiheadAttention(
            channels, n_heads, dropout=dropout, batch_first=True)
        self.norm2      = nn.LayerNorm(channels)
        self.linear1    = nn.Linear(channels, dim_feedforward)
        self.linear2    = nn.Linear(dim_feedforward, channels)
        self.dropout    = nn.Dropout(dropout)
        self.act        = nn.GELU()
        self.block_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x
        h = x.transpose(1, 2).contiguous()
        h = self.pe(h)

        normed = self.norm1(h)
        a, _ = self.self_attn(normed, normed, normed)
        h = h + self.dropout(a)

        h = h + self.dropout(
            self.linear2(self.dropout(self.act(self.linear1(self.norm2(h))))))

        h = h.transpose(1, 2).contiguous()
        return identity + torch.tanh(self.block_gate) * h

class PatchedTransformerBlock(nn.Module):
    pass
    def __init__(self, patch_size=10, patch_dim=16, n_heads=1,
                 dim_feedforward=64, dropout=0.1, max_patches=512):
        super().__init__()
        self.patch_size   = patch_size
        self.patch_embed  = nn.Conv1d(
            1, patch_dim, kernel_size=patch_size, stride=patch_size)
        self.pe           = PositionalEncoding(patch_dim, dropout, max_patches)
        self.norm1        = nn.LayerNorm(patch_dim)
        self.self_attn    = nn.MultiheadAttention(
            patch_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2        = nn.LayerNorm(patch_dim)
        self.linear1      = nn.Linear(patch_dim, dim_feedforward)
        self.linear2      = nn.Linear(dim_feedforward, patch_dim)
        self.dropout      = nn.Dropout(dropout)
        self.act          = nn.GELU()
        self.patch_unembed= nn.ConvTranspose1d(
            patch_dim, 1, kernel_size=patch_size, stride=patch_size)
        self.block_gate   = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x
        h = self.patch_embed(x)
        h = h.transpose(1, 2).contiguous()
        h = self.pe(h)

        normed = self.norm1(h)
        a, _ = self.self_attn(normed, normed, normed)
        h = h + self.dropout(a)

        h = h + self.dropout(
            self.linear2(self.dropout(self.act(self.linear1(self.norm2(h))))))

        h = h.transpose(1, 2).contiguous()
        h = self.patch_unembed(h)
        return identity + torch.tanh(self.block_gate) * h

class LeanCNN(nn.Module):
    def __init__(self, d_model=256, dropout=0.1):
        super().__init__()

        self.t0 = PatchedTransformerBlock(
            patch_size=10, patch_dim=16, n_heads=1,
            dim_feedforward=64, dropout=dropout, max_patches=512)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4), nn.ReLU())

        self.t1 = InterleavedTransformerBlock(
            channels=32, n_heads=2, dim_feedforward=128,
            dropout=dropout, max_len=768)
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1), nn.ReLU())

        self.t2 = InterleavedTransformerBlock(
            channels=64, n_heads=4, dim_feedforward=256,
            dropout=dropout, max_len=768)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2), nn.ReLU())

        self.t3 = InterleavedTransformerBlock(
            channels=128, n_heads=8, dim_feedforward=512,
            dropout=dropout, max_len=384)
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, d_model, kernel_size=5, stride=1), nn.ReLU())

        self.se = SEBlock1d(d_model, reduction=16)

    def forward(self, x):
        x = self.t0(x)
        x = self.conv1(x)
        x = self.t1(x)
        x = self.conv2(x)
        x = self.t2(x)
        x = self.conv3(x)
        x = self.t3(x)
        x = self.conv4(x)
        x = self.se(x)
        x = x.transpose(1, 2).contiguous()
        return x

class TransformerEncoderLayerWithAttnResidual(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        self.act       = nn.GELU()
        self.attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        normed = self.norm1(x)
        a, _ = self.self_attn(normed, normed, normed,
                               attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)
        del normed
        a = self.dropout(a)
        x = x + a
        x = x + self.dropout(
            self.linear2(self.dropout(self.act(self.linear1(self.norm2(x))))))
        x = x + torch.tanh(self.attn_gate) * a
        del a
        return x

class TransformerDecoderLayerWithAttnResidual(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn       = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn      = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1         = nn.Linear(d_model, dim_feedforward)
        self.linear2         = nn.Linear(dim_feedforward, d_model)
        self.norm1           = nn.LayerNorm(d_model)
        self.norm2           = nn.LayerNorm(d_model)
        self.norm3           = nn.LayerNorm(d_model)
        self.dropout         = nn.Dropout(dropout)
        self.act             = nn.GELU()
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, tgt, memory,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        normed1 = self.norm1(tgt)
        sa, _ = self.self_attn(normed1, normed1, normed1,
                                attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
        del normed1
        tgt = tgt + self.dropout(sa)
        del sa

        ca, _ = self.cross_attn(self.norm2(tgt), memory, memory,
                                 key_padding_mask=memory_key_padding_mask)
        ca  = self.dropout(ca)
        tgt = tgt + ca

        tgt = tgt + self.dropout(
            self.linear2(self.dropout(self.act(self.linear1(self.norm3(tgt))))))

        tgt = tgt + torch.tanh(self.cross_attn_gate) * ca
        del ca
        return tgt

class AttentionHead(nn.Module):
    def __init__(self, d_model, n_heads, dropout, vocab_size, bias=True):
        super().__init__()
        self.norm      = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout   = nn.Dropout(dropout)

        self.head_gate = nn.Parameter(torch.zeros(1))
        self.proj      = nn.Sequential(
            Linear(d_model, d_model, bias),
            nn.ReLU(),
            Linear(d_model, vocab_size, bias, init="final"),
        )

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        normed = self.norm(tgt)
        a, _ = self.self_attn(normed, normed, normed,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + torch.tanh(self.head_gate) * self.dropout(a)
        return self.proj(tgt)

class Model(nn.Module):
    def __init__(self, d_model, en_layers, de_layers, en_head, de_head,
                 en_dim_feed, de_dim_feed, dropout, max_len, vocab_size,
                 bias=True):
        super(Model, self).__init__()

        self.cnn = LeanCNN(d_model=d_model, dropout=dropout)

        self.encoder  = nn.ModuleList([
            TransformerEncoderLayerWithAttnResidual(
                d_model, en_head, en_dim_feed, dropout)
            for _ in range(en_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        self.decoder  = nn.ModuleList([
            TransformerDecoderLayerWithAttnResidual(
                d_model, de_head, de_dim_feed, dropout)
            for _ in range(de_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        self.enc_pe    = PositionalEncoding(d_model, dropout, max_len=512)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model)
        self.pe        = PositionalEncoding(d_model, dropout, max_len)

        self.pre_type  = AttentionHead(
            d_model, de_head, dropout, vocab_size, bias)

    def forward(self, en, de_1, tgt_mask, tgt_key_padding_mask):
        x = self.cnn(en)
        x = self.enc_pe(x)

        for layer in self.encoder:
            x = layer(x)
        x = self.enc_norm(x)

        tgt = self.pe(self.embedding(de_1))
        for layer in self.decoder:
            tgt = layer(tgt, x,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask)
        tgt = self.dec_norm(tgt)

        return self.pre_type(tgt,
                             tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask)

@torch.no_grad()
def beam_search_top1(model, spectrum, max_len=100, beam_width=10,
                     bos_idx=BOS_IDX, eos_idx=EOS_IDX):
    pass
    was_training = model.training
    model.eval()
    device = spectrum.device

    x = model.cnn(spectrum.unsqueeze(0))
    x = model.enc_pe(x)
    for layer in model.encoder:
        x = layer(x)
    memory = model.enc_norm(x)

    beams = [([bos_idx], 0.0)]
    finished = []

    for _ in range(max_len):
        cands = []
        for seq, score in beams:
            if seq[-1] == eos_idx:
                finished.append((seq, score))
                continue
            tgt = torch.tensor(seq, device=device).unsqueeze(0)
            h = model.pe(model.embedding(tgt))
            sub = subsequent_mask(len(seq)).to(device)
            for layer in model.decoder:
                h = layer(h, memory, tgt_mask=sub)
            h = model.dec_norm(h)

            head_logits = model.pre_type(h, tgt_mask=sub)
            log_probs = F.log_softmax(
                head_logits[:, -1, :].squeeze(0), dim=-1)
            top_lp, top_id = log_probs.topk(beam_width)
            for v, t in zip(top_lp.tolist(), top_id.tolist()):
                cands.append((seq + [t], score + v))
        if not cands:
            break
        cands.sort(key=lambda z: z[1], reverse=True)
        beams = cands[:beam_width]
        if all(b[0][-1] == eos_idx for b in beams):
            finished.extend(beams)
            break

    finished.extend(beams)
    finished.sort(key=lambda z: z[1], reverse=True)

    if was_training:
        model.train()
    return finished[0][0]

def unlikelihood_loss(model, spectrum, wrong_seq, gt_seq):
    pass
    device = spectrum.device

    if len(wrong_seq) < 3:
        return torch.tensor(0.0, device=device)

    div = min(len(wrong_seq), len(gt_seq))
    for i in range(1, min(len(wrong_seq), len(gt_seq))):
        if wrong_seq[i] != gt_seq[i]:
            div = i
            break

    de_in  = torch.tensor(wrong_seq[:-1], device=device).unsqueeze(0)
    labels = torch.tensor(wrong_seq[1:],  device=device).unsqueeze(0)
    pad    = padding_mask(de_in, pad_idx=PAD_IDX)
    sub    = subsequent_mask(de_in.size(1)).to(device)

    logits = model(spectrum.unsqueeze(0), de_in, sub, pad)
    probs  = F.softmax(logits, dim=-1)
    p_wrong = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    pos = torch.zeros_like(labels, dtype=torch.bool)
    pos[:, max(0, div - 1):] = True
    pos &= (labels != PAD_IDX)
    if not pos.any():
        return torch.tensor(0.0, device=device)

    ul = -torch.log(torch.clamp(1.0 - p_wrong, min=1e-8))
    return (ul * pos.float()).sum() / pos.float().sum()

device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs      = 200
d_model     = 256
en_layers   = 6
de_layers   = 6
en_head     = 8
de_head     = 8
en_dim_feed = 1024
de_dim_feed = 1024
dropout     = 0.1
max_len     = 100

UL_START_EPOCH = 30
UL_EVERY       = 10
LAMBDA_UL      = 0.2
BEAM_WIDTH     = 1

RESUME_FROM = "best_SE_AR_lean_AH.pt"
FINETUNE_LR = 5e-5

train_data = torch.load("qm9s_train_irdata.pt", weights_only=False)
valid_data = torch.load("qm9s_valid_irdata.pt", weights_only=False)

print(f"Training samples : {len(train_data)}")
print(f"Validation samples: {len(valid_data)}")

train_loader = DataLoader(train_data, batch_size=32, collate_fn=collate_fn,
                          pin_memory=True, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, collate_fn=collate_fn,
                          pin_memory=True, shuffle=True)

model = Model(
    d_model, en_layers, de_layers, en_head, de_head,
    en_dim_feed, de_dim_feed, dropout, max_len, vocab_size,
    bias=True,
).to(device)

loss_fn       = Loss(model)
loss_fn.to(device)
type_accuracy = TypeAccuracy(eos_idx=2)
type_accuracy.to(device)

if RESUME_FROM is not None:
    print(f"Loading baseline weights from '{RESUME_FROM}' for UL fine-tuning")
    state = torch.load(RESUME_FROM, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"  Fine-tune LR: {FINETUNE_LR} (warmup skipped, UL active from epoch 0)")
    initial_lr = FINETUNE_LR
else:
    print("Training from scratch (15-epoch warmup, UL active from epoch "
          f"{UL_START_EPOCH})")
    initial_lr = 2e-4

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=initial_lr,
    betas=(0.9, 0.99),
    weight_decay=5e-4
)

WARMUP_EPOCHS = 15

if RESUME_FROM is not None:

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 1.0)
else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / WARMUP_EPOCHS)
    )

def train_one_epoch(model, loader, optimizer, loss_fn, accuracy_fn, device,
                    ul_active=False):
    pass
    model.train()
    total_loss     = 0.0
    total_accuracy = 0
    total_ul_loss  = 0.0
    n_ul_updates   = 0
    n_ul_skipped   = 0

    for batch_index, (input_x, de_input, pad_mask, sub_mask, label_idx) in enumerate(loader):
        input_x   = input_x.to(device)
        de_input  = de_input.to(device)
        pad_mask  = pad_mask.to(device)
        sub_mask  = sub_mask.to(device)
        label_idx = label_idx.to(device)

        optimizer.zero_grad()
        loss, pred_types = loss_fn(input_x, de_input, sub_mask, pad_mask, label_idx)

        if ul_active and (batch_index + 1) % UL_EVERY == 0:
            i = torch.randint(0, input_x.size(0), (1,)).item()

            gt_lab = label_idx[i].tolist()
            if EOS_IDX in gt_lab:
                gt_lab = gt_lab[:gt_lab.index(EOS_IDX) + 1]
            gt = [BOS_IDX] + gt_lab

            wrong = beam_search_top1(
                model, input_x[i], max_len=max_len,
                beam_width=BEAM_WIDTH)

            if wrong != gt and len(wrong) >= 3:
                ul = unlikelihood_loss(model, input_x[i], wrong, gt)
                if ul.item() > 0.0:
                    loss = loss + LAMBDA_UL * ul
                    total_ul_loss += ul.item()
                    n_ul_updates  += 1
                else:
                    n_ul_skipped  += 1
            else:

                n_ul_skipped += 1

        loss.backward()
        optimizer.step()

        total_loss     += loss.item()
        accuracy        = accuracy_fn(pred_types, label_idx)[0]
        total_accuracy += accuracy

        if (batch_index + 1) % 10 == 0:
            ul_info = ""
            if ul_active:
                ul_info = (f" | UL applied: {n_ul_updates}, "
                           f"skipped: {n_ul_skipped}")
            print(f"  Step [{batch_index+1}/{len(loader)}], "
                  f"Loss: {total_loss / (batch_index+1):.4f}{ul_info}")

    avg_ul = (total_ul_loss / n_ul_updates) if n_ul_updates > 0 else 0.0
    return (total_loss / len(loader),
            total_accuracy / len(loader.dataset),
            avg_ul,
            n_ul_updates)

def validate(model, loader, loss_fn, accuracy_fn, device):
    model.eval()
    total_loss     = 0
    total_accuracy = 0
    with torch.no_grad():
        for input_x, de_input, pad_mask, sub_mask, label_idx in loader:
            input_x   = input_x.to(device)
            de_input  = de_input.to(device)
            pad_mask  = pad_mask.to(device)
            sub_mask  = sub_mask.to(device)
            label_idx = label_idx.to(device)

            loss, pred_types = loss_fn(
                input_x, de_input, sub_mask, pad_mask, label_idx)
            total_loss     += loss.item()
            accuracy, _, _  = accuracy_fn(pred_types, label_idx)
            total_accuracy += accuracy

    return total_loss / len(loader), total_accuracy / len(loader.dataset)

def main():
    max_accuracy    = 0
    best_model      = None
    counter         = 0
    best_valid_loss = float('inf')

    train_losses, valid_losses         = [], []
    train_accuracies, valid_accuracies = [], []
    ul_avg_losses, ul_counts           = [], []

    for epoch in range(epochs):
        start_time = time.time()

        ul_active  = True if RESUME_FROM is not None else (epoch >= UL_START_EPOCH)

        train_loss, train_accuracy, ul_avg, ul_n = train_one_epoch(
            model, train_loader, optimizer, loss_fn, type_accuracy, device,
            ul_active=ul_active)
        valid_loss, valid_accuracy = validate(
            model, valid_loader, loss_fn, type_accuracy, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        ul_avg_losses.append(ul_avg)
        ul_counts.append(ul_n)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - start_time
        ul_str = (f" | UL: avg={ul_avg:.4f} n={ul_n}"
                  if ul_active else " | UL: off (warmup)")
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f} | "
              f"Train Acc: {train_accuracy:.4f}  Valid Acc: {valid_accuracy:.4f} | "
              f"LR: {current_lr:.2e}{ul_str} | Time: {epoch_time:.2f}s")

        if valid_accuracy > max_accuracy:
            max_accuracy = valid_accuracy
            best_model   = copy.deepcopy(model)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
        else:
            counter += 1
            if counter >= 10:
                print("Early stopping triggered.")
                break

        if RESUME_FROM is None and (epoch + 1) == UL_START_EPOCH:
            print(f"  → UL activates next epoch. "
                  f"Resetting early-stopping counter and best_valid_loss.")
            counter         = 0
            best_valid_loss = float('inf')

    suffix = "_finetune" if RESUME_FROM is not None else ""
    ckpt_path = f"best_SE_AR_lean_UL_AH{suffix}.pt"
    torch.save(best_model.state_dict(), ckpt_path)
    print(f"Saved best model to {ckpt_path}")

    epochs_range = range(1, len(train_losses) + 1)
    show_ul_vline = (RESUME_FROM is None) and any(c > 0 for c in ul_counts)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, valid_losses, label='Valid Loss')
    if show_ul_vline:
        plt.axvline(x=UL_START_EPOCH, color='gray', linestyle='--',
                    label=f'UL starts (epoch {UL_START_EPOCH})')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    title_suffix = " (fine-tune)" if RESUME_FROM is not None else ""
    plt.title(f'Loss over Epochs — Lean variant + UL + attn head{title_suffix}')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"loss_curve_LA_SE_AR_lean_UL_AH{suffix}.png"); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, valid_accuracies, label='Valid Accuracy')
    if show_ul_vline:
        plt.axvline(x=UL_START_EPOCH, color='gray', linestyle='--',
                    label=f'UL starts (epoch {UL_START_EPOCH})')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title(f'Accuracy over Epochs — Lean variant + UL + attn head{title_suffix}')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"accuracy_curve_LA_SE_AR_lean_UL_AH{suffix}.png"); plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs_range, ul_avg_losses, color='tab:red', label='Avg UL Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Avg UL Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.plot(epochs_range, ul_counts, color='tab:blue',
             label='# UL updates', alpha=0.6)
    ax2.set_ylabel('# UL updates / epoch', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    if show_ul_vline:
        plt.axvline(x=UL_START_EPOCH, color='gray', linestyle='--', alpha=0.6)
    plt.title(f'UL diagnostics over Epochs{title_suffix}')
    fig.tight_layout()
    plt.savefig(f"ul_curve_LA_SE_AR_lean_UL_AH{suffix}.png"); plt.close()

if __name__ == "__main__":
    main()
