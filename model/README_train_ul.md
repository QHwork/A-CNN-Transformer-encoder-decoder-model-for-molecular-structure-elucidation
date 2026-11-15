# model_train_ul.py

From `model/`, run `python model_train_ul.py` with:

- `qm9s_train_irdata.pt`
- `qm9s_valid_irdata.pt`
- `best_SE_AR_lean_AH.pt`

There are no command-line arguments. The default `RESUME_FROM` loads the Base
checkpoint and performs unlikelihood fine-tuning with learning rate 5e-5, a fresh
AdamW optimizer and no learning-rate warmup. UL is active immediately in this
fine-tuning mode. Architecture, vocabulary and dataset format match Base.
The unlikelihood weight is 0.2 and the UL term is applied every 10 batches.

The default checkpoint output is `best_SE_AR_lean_UL_AH_finetune.pt`. Training
also writes plots. Files are written into the current working directory and can
overwrite previous outputs. To train from scratch, set `RESUME_FROM = None`;
this uses the separate scratch schedule and writes `best_SE_AR_lean_UL_AH.pt`.

Dependencies: PyTorch, NumPy and Matplotlib. CUDA is selected when available.
This copy preserves the original UL implementation. It is distinct from the
separate softmax-fix experiment and is provided for the historical Base/UL workflow.
