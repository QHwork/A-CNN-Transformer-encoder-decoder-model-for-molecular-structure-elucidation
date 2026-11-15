# model_build_dataset.py

From `model/`, run `python model_build_dataset.py`.

Inputs (relative to the current working directory):

- `original_data/qm9s_SMILES_native_kekule.txt`: one native Kekule SMILES per line.
- `original_data/qm9s_irdata.csv`: header row, ID column, then IR intensity columns.

The builder removes the CSV header and ID column, keeps the original intensity
grid, tokenizes SMILES with the fixed 16-token vocabulary and rejects unsupported
tokens. Output spectrum shape is `(N, 1, T)` with float32 values.

For each spectrum, the original implementation sets positions equal to its
minimum to zero and scales the remaining positions as `100 * y / y_max`.
A zero maximum uses denominator 1. This is not the usual formula
`100 * (y - y_min) / (y_max - y_min)` when the original minimum is nonzero.
The implementation is preserved to match the original dataset workflow.

Zero-based source row index modulo 10 defines the fixed split: 0–7 train, 8
validation, 9 test. For 127,468 aligned rows the counts are 101,976 / 12,746 / 12,746.

Outputs in the working directory:

- `qm9s_train_irdata.pt`
- `qm9s_valid_irdata.pt`
- `qm9s_test_irdata.pt`

Dependencies: NumPy and PyTorch. Outputs are serialized `MyDataSet` objects; the
training/evaluation scripts contain the compatibility class needed to read them.
