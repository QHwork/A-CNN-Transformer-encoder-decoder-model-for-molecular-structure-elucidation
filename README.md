# QM9s IR model scripts

Run the commands below from the repository root, where this README and
`requirements.txt` are located. The scripts and their documentation are in
`model/`; pretrained checkpoints are in `saved_pt_files/`. These standalone scripts
correspond to the original Base/UL workflow used by this website. They retain the
original training and evaluation logic; they do not include the separately
developed softmax-fix training variant. Comments and docstrings have been removed;
usage documentation is kept in the README files instead.

| Script | Documentation | Purpose | Saved dataset / checkpoint files |
| --- | --- | --- | --- |
| [model_build_dataset.py](model/model_build_dataset.py) | [README_build_dataset.md](model/README_build_dataset.md) | Build IR train/validation/test datasets | `qm9s_train_irdata.pt`, `qm9s_valid_irdata.pt`, `qm9s_test_irdata.pt` (download link below) |
| [model_train_base.py](model/model_train_base.py) | [README_train_base.md](model/README_train_base.md) | Train the Base CNN-Transformer | [best_SE_AR_lean_AH.pt](saved_pt_files/best_SE_AR_lean_AH.pt) |
| [model_train_ul.py](model/model_train_ul.py) | [README_train_ul.md](model/README_train_ul.md) | Fine-tune Base with unlikelihood loss | [best_SE_AR_lean_UL_AH_finetune.pt](saved_pt_files/best_SE_AR_lean_UL_AH_finetune.pt) |
| [model_evaluate.py](model/model_evaluate.py) | [README_evaluate.md](model/README_evaluate.md) | Evaluate either checkpoint with Greedy and Beam Top-1/5/10 | Reads either checkpoint above; saves evaluation results with `--out-file` |

## Environment

Use Python 3.11 and a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows use `.venv\Scripts\activate` instead. For a CUDA server, install the
PyTorch build matching the server's CUDA support using the official
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Training and evaluation choose CUDA when available and otherwise use CPU.
The pinned versions describe the packaging environment; a full cloud training
run has not been performed for these renamed copies.

## End-to-end workflow

### Evaluate the supplied checkpoints

Download the original split files using the Google Drive link below and place
`qm9s_train_irdata.pt`, `qm9s_valid_irdata.pt` and `qm9s_test_irdata.pt` in `Data/`.
From the repository root, evaluate the supplied checkpoints directly:

```bash
python model/model_evaluate.py --checkpoint saved_pt_files/best_SE_AR_lean_AH.pt --test-data Data/qm9s_test_irdata.pt --greedy --beam --beam-width 10 --out-file base_results.txt
python model/model_evaluate.py --checkpoint saved_pt_files/best_SE_AR_lean_UL_AH_finetune.pt --test-data Data/qm9s_test_irdata.pt --greedy --beam --beam-width 10 --out-file ul_results.txt
```

### Build datasets and train new checkpoints

The original scripts resolve input and output paths relative to the working
directory. When running from the repository root, place the raw inputs at:

```text
original_data/qm9s_SMILES_native_kekule.txt
original_data/qm9s_irdata.csv
```

```bash
python model/model_build_dataset.py
python model/model_train_base.py
python model/model_train_ul.py
```

The builder writes the three dataset `.pt` files to the repository root, where
both training scripts expect them. If using downloaded splits instead of running
the builder, copy the three `.pt` files from `Data/` to the repository root first.
Base training writes `best_SE_AR_lean_AH.pt` to the working directory; UL training
loads that file and writes `best_SE_AR_lean_UL_AH_finetune.pt` there. These new
outputs are separate from the supplied checkpoints in `saved_pt_files/`.
To fine-tune the supplied Base checkpoint directly, copy
`saved_pt_files/best_SE_AR_lean_AH.pt` to the repository root before running UL.
UL scratch mode instead saves `best_SE_AR_lean_UL_AH.pt`.

Training scripts load datasets at module startup: execute them directly rather
than importing them. Preserve previous outputs before repeating a run.
Only load trusted `.pt` datasets because dataset loading uses Python pickle.

## Data and benchmark

The original training, validation, and test split files are available for download
from [Google Drive](https://drive.google.com/drive/folders/1n6aCHxIgGofGnd-stYUY9BIV6VxjPuMT?usp=sharing).

QM9s IR contains 127,468 molecules: train 101,976, validation 12,746 and test 12,746.
The split is fixed by zero-based source row index modulo 10: 0–7 train, 8 valid,
9 test. The original 3000 spectral values are retained.

| Model | Saved checkpoint | Greedy | Beam Top-1 | Beam Top-5 | Beam Top-10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | [best_SE_AR_lean_AH.pt](saved_pt_files/best_SE_AR_lean_AH.pt) | 66.03% | 67.05% | 89.30% | 93.25% |
| UL | [best_SE_AR_lean_UL_AH_finetune.pt](saved_pt_files/best_SE_AR_lean_UL_AH_finetune.pt) | 72.57% | 72.91% | 91.31% | 94.58% |

These are reported results from defense slide 26, not results newly reproduced
by this packaging step. The builder's exact normalization is documented in
[model/README_build_dataset.md](model/README_build_dataset.md). The website's upload resampling and normalization are
a separate browser inference path.

## Website

The interactive QM9s IR-to-SMILES demo is available at
[https://website-qm-9s-model.vercel.app](https://website-qm-9s-model.vercel.app).

## Related article

The related article is available at
[https://doi.org/10.21203/rs.3.rs-10731198/v1](https://doi.org/10.21203/rs.3.rs-10731198/v1).

Qinghai Cui is the first author and sole corresponding author of this article.
