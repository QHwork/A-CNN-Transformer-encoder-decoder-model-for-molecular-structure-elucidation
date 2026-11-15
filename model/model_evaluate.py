pass

import argparse
import sys
import types
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

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
id_to_token = {v: k for k, v in dic.items()}

def ids_to_smiles(ids):
    pass
    tokens = []
    for i in ids:
        if i == BOS_IDX:
            continue
        if i == EOS_IDX or i == PAD_IDX:
            break
        tokens.append(id_to_token.get(i, '?'))
    return ''.join(tokens)

def extract_gt(sample_y):
    pass
    gt = list(sample_y)
    if gt[0] != BOS_IDX:
        gt = [BOS_IDX] + gt
    if EOS_IDX in gt:
        gt = gt[:gt.index(EOS_IDX) + 1]
    while gt and gt[-1] == PAD_IDX:
        gt.pop()
    return gt

def subsequent_mask(size):
    return torch.from_numpy(
        np.triu(np.ones((size, size)), k=1).astype('uint8')) == 1

_MASK_CACHE = {}

def get_subsequent_mask(size, device):
    cached = _MASK_CACHE.get(device)
    if cached is None or cached.size(0) < size:
        cached = subsequent_mask(size).to(device)
        _MASK_CACHE[device] = cached
    return cached[:size, :size]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
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
    def __init__(self, d_in, d_out, bias=True, init="default"):
        super().__init__(d_in, d_out, bias=bias)
        self.use_bias = bias
        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)
        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "final":
            self._zero_init(False)

    def _trunc_normal_init(self, scale=1.0):
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale ** 0.5) / 0.87962566103423978
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
        self.excite = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        avg = self.avg_pool(x).squeeze(-1)
        mx = self.max_pool(x).squeeze(-1)
        scale = torch.sigmoid(self.excite(avg) + self.excite(mx))
        return x * scale.unsqueeze(-1)

class InterleavedTransformerBlock(nn.Module):
    def __init__(self, channels, n_heads, dim_feedforward, dropout, max_len):
        super().__init__()
        self.pe = PositionalEncoding(channels, dropout, max_len)
        self.norm1 = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(
            channels, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
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
    def __init__(self, patch_size=10, patch_dim=16, n_heads=1,
                 dim_feedforward=64, dropout=0.1, max_patches=512):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv1d(
            1, patch_dim, kernel_size=patch_size, stride=patch_size)
        self.pe = PositionalEncoding(patch_dim, dropout, max_patches)
        self.norm1 = nn.LayerNorm(patch_dim)
        self.self_attn = nn.MultiheadAttention(
            patch_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(patch_dim)
        self.linear1 = nn.Linear(patch_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, patch_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.patch_unembed = nn.ConvTranspose1d(
            patch_dim, 1, kernel_size=patch_size, stride=patch_size)
        self.block_gate = nn.Parameter(torch.zeros(1))

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
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
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
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, tgt, memory, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        normed1 = self.norm1(tgt)
        sa, _ = self.self_attn(normed1, normed1, normed1,
                               attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask)
        del normed1
        tgt = tgt + self.dropout(sa)
        del sa
        ca, _ = self.cross_attn(self.norm2(tgt), memory, memory,
                                key_padding_mask=memory_key_padding_mask)
        ca = self.dropout(ca)
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
        super().__init__()
        self.cnn = LeanCNN(d_model=d_model, dropout=dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoderLayerWithAttnResidual(
                d_model, en_head, en_dim_feed, dropout)
            for _ in range(en_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList([
            TransformerDecoderLayerWithAttnResidual(
                d_model, de_head, de_dim_feed, dropout)
            for _ in range(de_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.enc_pe = PositionalEncoding(d_model, dropout, max_len=512)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model, dropout, max_len)
        self.pre_type = AttentionHead(
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

def _spectrum_tensor(spectrum_raw):
    pass
    return torch.as_tensor(spectrum_raw, dtype=torch.float32)

@torch.inference_mode()
def encode_spectra_batch(model, spectra):
    pass
    x = model.cnn(spectra)
    x = model.enc_pe(x)
    for layer in model.encoder:
        x = layer(x)
    return model.enc_norm(x)

@torch.inference_mode()
def decode_step_batch(model, memory, seqs, sample_idx, device):
    pass
    N = len(seqs)
    lengths = [len(s) for s in seqs]
    T = max(lengths)

    tgt = torch.full((N, T), PAD_IDX, dtype=torch.long, device=device)
    for k, s in enumerate(seqs):
        tgt[k, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    h = model.pe(model.embedding(tgt))
    sub = get_subsequent_mask(T, device)
    tgt_kpm = (tgt == PAD_IDX)
    if not bool(tgt_kpm.any()):
        tgt_kpm = None

    mem = memory.index_select(
        0, torch.tensor(sample_idx, dtype=torch.long, device=device))

    for layer in model.decoder:
        h = layer(h, mem, tgt_mask=sub, tgt_key_padding_mask=tgt_kpm)
    h = model.dec_norm(h)

    head_logits = model.pre_type(h, tgt_mask=sub, tgt_key_padding_mask=tgt_kpm)

    last = head_logits[torch.arange(N, device=device),
                       torch.tensor(lengths, dtype=torch.long, device=device) - 1]

    return F.log_softmax(last, dim=-1)

@torch.inference_mode()
def beam_search_batch(model, memory, beam_width=10, max_len=100, device='cuda'):
    pass
    B = memory.size(0)
    beams = [[([BOS_IDX], 0.0)] for _ in range(B)]
    finished = [[] for _ in range(B)]
    done = [False] * B

    for _ in range(max_len):
        process_order = []
        active_per_sample = {}
        flat_seqs, flat_sample = [], []
        for i in range(B):
            if done[i]:
                continue
            active = []
            for seq, score in beams[i]:
                if seq[-1] == EOS_IDX:
                    finished[i].append((seq, score))
                else:
                    active.append((seq, score))
            if not active:
                done[i] = True
                continue
            active_per_sample[i] = active
            process_order.append(i)
            for seq, score in active:
                flat_seqs.append(seq)
                flat_sample.append(i)

        if not flat_seqs:
            break

        log_probs = decode_step_batch(model, memory, flat_seqs, flat_sample, device)
        top_lp, top_id = log_probs.topk(beam_width, dim=-1)
        top_lp = top_lp.tolist()
        top_id = top_id.tolist()

        row = 0
        for i in process_order:
            cands = []
            for (seq, score) in active_per_sample[i]:
                vals = top_lp[row]
                toks = top_id[row]
                row += 1
                for v, t in zip(vals, toks):
                    cands.append((seq + [t], score + v))
            cands.sort(key=lambda z: z[1], reverse=True)
            beams[i] = cands[:beam_width]
            if all(b[0][-1] == EOS_IDX for b in beams[i]):
                finished[i].extend(beams[i])
                done[i] = True

    results = []
    for i in range(B):
        finished[i].extend(beams[i])
        finished[i].sort(key=lambda z: z[1], reverse=True)
        results.append(finished[i][:beam_width])
    return results

@torch.inference_mode()
def greedy_decode_batch(model, memory, max_len=100, device='cuda'):
    pass
    B = memory.size(0)
    seqs = [[BOS_IDX] for _ in range(B)]
    done = [False] * B

    for _ in range(max_len):
        order, flat_seqs, flat_sample = [], [], []
        for i in range(B):
            if done[i]:
                continue
            order.append(i)
            flat_seqs.append(seqs[i])
            flat_sample.append(i)
        if not flat_seqs:
            break

        log_probs = decode_step_batch(model, memory, flat_seqs, flat_sample, device)
        nxt = log_probs.argmax(dim=-1).tolist()
        for k, i in enumerate(order):
            t = nxt[k]
            seqs[i].append(t)
            if t == EOS_IDX:
                done[i] = True

    return seqs

def _iter_batches(n, batch_size):
    for start in range(0, n, batch_size):
        yield list(range(start, min(start + batch_size, n)))

def evaluate_greedy_batched(model, test_data, batch_size, device):
    model.eval()
    n = len(test_data)
    correct = 0
    wrongs = []
    t0 = time.time()
    seen = 0
    for idxs in _iter_batches(n, batch_size):
        spectra = torch.stack(
            [_spectrum_tensor(test_data[i][0]) for i in idxs]).to(device)
        gts = [extract_gt(test_data[i][1]) for i in idxs]
        memory = encode_spectra_batch(model, spectra)
        preds = greedy_decode_batch(model, memory, max_len=100, device=device)
        for j, i in enumerate(idxs):
            if preds[j] == gts[j]:
                correct += 1
            else:
                wrongs.append((i, ids_to_smiles(gts[j]), ids_to_smiles(preds[j])))
        seen += len(idxs)
        elapsed = time.time() - t0
        rate = seen / elapsed if elapsed > 0 else 0
        eta = (n - seen) / rate if rate > 0 else 0
        print(f"  [{seen:>6}/{n}]  acc so far: {correct/seen:.4f}  "
              f"({rate:.1f} samples/s, ETA: {eta:.0f}s)")
    return correct / n, wrongs

def evaluate_beam_batched(model, test_data, beam_width, batch_size, device):
    model.eval()
    n = len(test_data)
    top1 = top5 = top10 = 0
    wrongs = []
    t0 = time.time()
    seen = 0
    for idxs in _iter_batches(n, batch_size):
        spectra = torch.stack(
            [_spectrum_tensor(test_data[i][0]) for i in idxs]).to(device)
        gts = [extract_gt(test_data[i][1]) for i in idxs]
        memory = encode_spectra_batch(model, spectra)
        results = beam_search_batch(
            model, memory, beam_width=beam_width, max_len=100, device=device)
        for j, i in enumerate(idxs):
            seqs = [c[0] for c in results[j]]
            gt = gts[j]
            if seqs and seqs[0] == gt:
                top1 += 1
            if gt in seqs[:5]:
                top5 += 1
            if gt in seqs[:10]:
                top10 += 1
            if gt not in seqs:
                t1smi = ids_to_smiles(seqs[0]) if seqs else '<empty>'
                wrongs.append((i, ids_to_smiles(gt), t1smi))
        seen += len(idxs)
        elapsed = time.time() - t0
        rate = seen / elapsed if elapsed > 0 else 0
        eta = (n - seen) / rate if rate > 0 else 0
        print(f"  [{seen:>6}/{n}]  top1: {top1/seen:.4f}  "
              f"top5: {top5/seen:.4f}  top10: {top10/seen:.4f}  "
              f"({rate:.1f} samples/s, ETA: {eta:.0f}s)")
    return top1 / n, top5 / n, top10 / n, wrongs

def _show_wrongs(k, wrongs, col, logger=print):
    if k > 0 and wrongs:
        logger(f"\n  Showing {min(k, len(wrongs))} of {len(wrongs)} wrong:")
        logger(f"  {'Idx':>6}  {'Ground Truth':<30}  {col:<30}")
        logger(f"  {'---':>6}  {'---':<30}  {'---':<30}")
        for idx, gt_smi, pred_smi in wrongs[:k]:
            logger(f"  {idx:>6}  {gt_smi:<30}  {pred_smi:<30}")

def main():
    parser = argparse.ArgumentParser(
        description="Cross-sample batched evaluation of the model")
    parser.add_argument('--greedy', action='store_true',
                        help='Run batched greedy decoding')
    parser.add_argument('--beam', action='store_true',
                        help='Run batched beam search (top-1/5/10 accuracy)')
    parser.add_argument('--checkpoint', default='best_SE_AR_lean_UL_AH_finetune.pt')
    parser.add_argument('--test-data', default='qm9s_test_irdata.pt')
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='samples per batch (the decoder runs '
                             'batch_size x beam_width sequences/step; '
                             'lower to 32/16 if you hit OOM on a 16 GB GPU)')
    parser.add_argument('--show-wrongs', type=int, default=20)
    parser.add_argument('--out-file', type=str, default=None,
                        help='Optional text file to save the final evaluation results')
    args = parser.parse_args()

    if not args.greedy and not args.beam:
        parser.error("Specify at least one of --greedy or --beam")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading model from '{args.checkpoint}' ...")
    model = Model(
        d_model=256, en_layers=6, de_layers=6, en_head=8, de_head=8,
        en_dim_feed=1024, de_dim_feed=1024, dropout=0.1, max_len=100,
        vocab_size=vocab_size, bias=True,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters loaded.")

    print(f"Loading test data from '{args.test_data}' ...")
    test_data = torch.load(args.test_data, weights_only=False)
    print(f"  {len(test_data):,} test samples.  batch_size={args.batch_size}")

    output_logs = []
    def log(msg=""):
        print(msg)
        output_logs.append(msg)

    if args.greedy:
        log(f"\n{'='*60}\nGREEDY DECODING (batched)\n{'='*60}")
        acc, wrongs = evaluate_greedy_batched(
            model, test_data, args.batch_size, device)
        N = len(test_data)
        log(f"\n  Greedy exact-match accuracy: {acc:.4f} ({int(acc*N)}/{N})")
        _show_wrongs(args.show_wrongs, wrongs, 'Predicted', logger=log)

    if args.beam:
        log(f"\n{'='*60}\nBEAM SEARCH (batched, width={args.beam_width})\n{'='*60}")
        top1, top5, top10, wrongs = evaluate_beam_batched(
            model, test_data, args.beam_width, args.batch_size, device)
        N = len(test_data)
        log(f"\n  Top-1  exact-match accuracy: {top1:.4f} ({int(top1*N)}/{N})")
        log(f"  Top-5  exact-match accuracy: {top5:.4f} ({int(top5*N)}/{N})")
        log(f"  Top-10 exact-match accuracy: {top10:.4f} ({int(top10*N)}/{N})")
        _show_wrongs(args.show_wrongs, wrongs, 'Beam Top-1', logger=log)

    if args.out_file:
        with open(args.out_file, 'w') as f:
            f.write("\n".join(output_logs) + "\n")
        print(f"\nSaved final results to {args.out_file}")

    print("\nDone.")

if __name__ == '__main__':
    main()
