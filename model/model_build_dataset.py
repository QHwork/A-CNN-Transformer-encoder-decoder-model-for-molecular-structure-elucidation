pass

import csv
import gc
import re
import sys
import types

import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

fake_util = types.ModuleType("util")
fake_util.MyDataSet = MyDataSet
sys.modules["util"] = fake_util

HARDCODED_DIC = {
    "PAD": 0, "BOS": 1, "EOS": 2,
    "C": 3, "N": 4, "O": 5, "F": 6,
    "1": 7, "2": 8, "3": 9, "4": 10, "5": 11,
    "=": 12, "#": 13, "(": 14, ")": 15,
}

def smi_tokenizer(smi: str):
    pass
    pattern = (
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
        r"|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )
    tokens = re.findall(pattern, smi)
    assert smi == "".join(tokens), f"Tokenisation mismatch for: {smi!r}"
    return tokens

def load_spectrum_csv(path: str) -> torch.Tensor:
    pass
    with open(path, "r") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        T = len(header) - 1

    with open(path, "rb") as fh:
        n_rows = sum(1 for _ in fh) - 1

    out = np.empty((n_rows, T), dtype=np.float32)

    with open(path, "r") as fh:
        reader = csv.reader(fh)
        next(reader)
        for i, row in enumerate(reader):
            out[i] = np.asarray(row[1:], dtype=np.float32)

    return torch.from_numpy(out).unsqueeze(1)

def normalize_spectrum(spectrum_data: torch.Tensor) -> torch.Tensor:
    pass

    y_min = spectrum_data.amin(dim=-1, keepdim=True)
    y_max = spectrum_data.amax(dim=-1, keepdim=True)

    mask_gt_min = spectrum_data > y_min
    denom = torch.where(y_max != 0, y_max, torch.ones_like(y_max))

    spectrum_data.div_(denom).mul_(100.0)

    spectrum_data.masked_fill_(~mask_gt_min, 0.0)

    return spectrum_data

def build(
    smiles_path: str = "original_data/qm9s_SMILES_native_kekule.txt",
    ir_path:    str = "original_data/qm9s_irdata.csv",
    dictionary: dict = None,
):

    if dictionary is None:
        dic = dict(HARDCODED_DIC)
        print(f"  Using HARDCODED_DIC ({len(dic)} tokens) — matches the model.")
    else:
        dic = dictionary
        print(f"  Using caller-supplied dictionary ({len(dic)} tokens).")

    print(f"Reading SMILES from '{smiles_path}' …")
    with open(smiles_path, "r") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    BOS, EOS = "BOS", "EOS"
    if BOS not in dic or EOS not in dic:
        raise ValueError("Dictionary must contain 'BOS' and 'EOS' entries.")

    atoms_smiles_idxs = []
    bad_lines = []
    for lineno, smi in enumerate(lines, start=1):
        tokens = smi_tokenizer(smi)
        try:
            ids = [dic[BOS]] + [dic[tok] for tok in tokens] + [dic[EOS]]
        except KeyError as e:
            bad_lines.append((lineno, smi, e.args[0]))
            continue
        atoms_smiles_idxs.append(ids)

    if bad_lines:
        sample = "\n".join(
            f"  line {ln}: {smi!r}  ← unknown token {tok!r}"
            for ln, smi, tok in bad_lines[:10]
        )
        raise ValueError(
            f"{len(bad_lines)} SMILES contain tokens outside vocabulary:\n{sample}"
        )

    print(f"  {len(atoms_smiles_idxs):,} SMILES tokenised successfully.")

    print(f"Reading IR spectra from '{ir_path}' …")
    ir_tensor = load_spectrum_csv(ir_path)
    print(f"  {ir_tensor.shape[0]:,} IR spectra loaded, shape {tuple(ir_tensor.shape)}.")

    print("Normalising IR spectra per-spectrum (min -> 0, max -> 100) …")
    spectrum = normalize_spectrum(ir_tensor).contiguous()
    del ir_tensor
    gc.collect()
    print(f"  Processed IR tensor shape: {tuple(spectrum.shape)} (N, 1, T)")

    n = spectrum.shape[0]
    assert n == len(atoms_smiles_idxs), (
        f"Mismatch: {n} spectra vs {len(atoms_smiles_idxs)} SMILES"
    )

    idx = torch.arange(n)
    mod = idx % 10
    train_idx = idx[mod < 8]
    valid_idx = idx[mod == 8]
    test_idx  = idx[mod == 9]

    print(
        f"  Split: {train_idx.numel():,} train | "
        f"{valid_idx.numel():,} valid | {test_idx.numel():,} test"
    )

    def _save_split(split_idx: torch.Tensor, out_path: str, label: str):
        spec = spectrum.index_select(0, split_idx)
        smi  = [atoms_smiles_idxs[i] for i in split_idx.tolist()]
        ds   = MyDataSet(spec, smi)
        torch.save(ds, out_path)
        print(f"Saved {out_path}  ({label}: {len(smi):,} samples, spec shape {tuple(spec.shape)})")
        del spec, smi, ds
        gc.collect()

    _save_split(valid_idx, "qm9s_valid_irdata.pt", "valid")
    _save_split(test_idx,  "qm9s_test_irdata.pt",  "test")
    _save_split(train_idx, "qm9s_train_irdata.pt", "train")

    del spectrum, atoms_smiles_idxs
    gc.collect()

    return dic

if __name__ == "__main__":
    dic = build(
        smiles_path="original_data/qm9s_SMILES_native_kekule.txt",
        ir_path="original_data/qm9s_irdata.csv",
    )
    print(f"\nFinal dictionary ({len(dic)} tokens):")
    print(dic)
    assert dic == HARDCODED_DIC, (
        "Saved tokens will not align with the model's hardcoded vocabulary."
    )
    print("Vocabulary matches the model's hardcoded dictionary.")
