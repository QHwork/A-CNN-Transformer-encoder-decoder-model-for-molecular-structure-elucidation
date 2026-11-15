# model_train_base.py

From `model/`, run `python model_train_base.py` after building or downloading the
datasets. Inputs are `qm9s_train_irdata.pt` and `qm9s_valid_irdata.pt` in the current
working directory. No command-line arguments are defined; configuration variables
are in the script.

The original CNN-Transformer uses dimension 256, six encoder and six decoder
layers, eight attention heads, feed-forward dimension 1024, dropout 0.1 and a
16-token vocabulary. Training uses batch size 32, AdamW with learning rate 2e-4,
betas (0.9, 0.99), weight decay 5e-4 and a 15-epoch linear warmup. The epoch limit
is 200 with early stopping. Train and validation loaders shuffle reading order;
dataset membership stays fixed.

The best checkpoint is saved as `best_SE_AR_lean_AH.pt`. Training also writes
loss/accuracy plots. Existing outputs can be overwritten; preserve previous runs
before rerunning. The checkpoint contains the model state dictionary and is the
default starting point for `model_train_ul.py`.

Dependencies: PyTorch, NumPy and Matplotlib. CUDA is selected when available.
This is the original loss implementation matching the historical model workflow.
