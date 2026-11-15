# QM9s IR model scripts

Run the commands below from this `model/` directory. These four standalone scripts
correspond to the original Base/UL workflow used by this website. They retain the
original training and evaluation logic; they do not include the separately
developed softmax-fix training variant. Comments and docstrings have been removed;
usage documentation is kept in the README files instead.

| Script | Documentation | Purpose |
| --- | --- | --- |
| [model_build_dataset.py](model_build_dataset.py) | [README_build_dataset.md](README_build_dataset.md) | Build IR train/validation/test datasets |
| [model_train_base.py](model_train_base.py) | [README_train_base.md](README_train_base.md) | Train the Base CNN-Transformer |
| [model_train_ul.py](model_train_ul.py) | [README_train_ul.md](README_train_ul.md) | Fine-tune Base with unlikelihood loss |
| [model_evaluate.py](model_evaluate.py) | [README_evaluate.md](README_evaluate.md) | Evaluate either checkpoint with Greedy and Beam Top-1/5/10 |

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

Place the original input files under `model/original_data/`:

```text
original_data/qm9s_SMILES_native_kekule.txt
original_data/qm9s_irdata.csv
```

```bash
python model_build_dataset.py
python model_train_base.py
python model_train_ul.py
python model_evaluate.py --checkpoint best_SE_AR_lean_AH.pt --greedy --beam --beam-width 10 --out-file base_results.txt
python model_evaluate.py --checkpoint best_SE_AR_lean_UL_AH_finetune.pt --greedy --beam --beam-width 10 --out-file ul_results.txt
```

Training scripts load datasets at module startup: execute them directly rather
than importing them. They write checkpoints and plots into the working directory;
use separate working directories or back up outputs before repeating a run.
Only load trusted `.pt` datasets because dataset loading uses Python pickle.

## Data and benchmark

The original training, validation, and test split files are available for download
from [Google Drive](https://drive.google.com/drive/folders/1n6aCHxIgGofGnd-stYUY9BIV6VxjPuMT?usp=sharing).

QM9s IR contains 127,468 molecules: train 101,976, validation 12,746 and test 12,746.
The split is fixed by zero-based source row index modulo 10: 0–7 train, 8 valid,
9 test. The original 3000 spectral values are retained.

| Model | Greedy | Beam Top-1 | Beam Top-5 | Beam Top-10 |
| --- | ---: | ---: | ---: | ---: |
| Base | 66.03% | 67.05% | 89.30% | 93.25% |
| UL | 72.57% | 72.91% | 91.31% | 94.58% |

These are reported results from defense slide 26, not results newly reproduced
by this packaging step. The builder's exact normalization is documented in
README_build_dataset.md. The website's upload resampling and normalization are
a separate browser inference path.

## Website

The interactive QM9s IR-to-SMILES demo is available at
[https://website-qm-9s-model.vercel.app](https://website-qm-9s-model.vercel.app).

## Related article

The related article is available at
[https://doi.org/10.21203/rs.3.rs-10731198/v1](https://doi.org/10.21203/rs.3.rs-10731198/v1).

Qinghai Cui is the first author and sole corresponding author of this article.
