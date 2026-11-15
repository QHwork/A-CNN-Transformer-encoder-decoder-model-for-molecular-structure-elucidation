# model_evaluate.py

One evaluator supports both Base and UL checkpoints. Run from `model/`:

```bash
python model_evaluate.py --checkpoint best_SE_AR_lean_AH.pt --test-data qm9s_test_irdata.pt --greedy --beam --beam-width 10 --batch-size 64 --out-file base_results.txt
python model_evaluate.py --checkpoint best_SE_AR_lean_UL_AH_finetune.pt --test-data qm9s_test_irdata.pt --greedy --beam --beam-width 10 --batch-size 64 --out-file ul_results.txt
```

The evaluator reports exact token-sequence matches for Greedy and Beam Top-1,
Top-5 and Top-10. Top-1/5/10 are ranks from the same width-10 beam search; they
do not mean three separate runs with beam widths 1, 5 and 10. Reduce batch size
if GPU memory is insufficient. `--show-wrongs` controls printed error examples.
At least one of `--greedy` or `--beam` must be supplied.

Defaults: UL fine-tuned checkpoint, `qm9s_test_irdata.pt`, beam width 10, batch
size 64. CUDA is selected when available; otherwise CPU is used. The checkpoint
must contain a compatible original model state dictionary. Dependencies are
PyTorch and NumPy. `python model_evaluate.py --help` lists all options.
