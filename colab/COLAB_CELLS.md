# Colab cells (copy each block into its own cell, in order)

Drive layout you create once (in Google Drive, "My Drive"):
```
logagent/data/processed/   <- upload train.jsonl, val.jsonl, val_eval.jsonl, test_eval.jsonl (and pilot\ folder, optional test.jsonl)
```
Outputs are written to `logagent/pilot_run`, `logagent/run1`, `logagent/predictions`, `logagent/merged`.

## A. Setup (run every time the runtime is new)

Cell 1 - check the GPU (Runtime > Change runtime type > T4 GPU first):
```python
!nvidia-smi
```
Cell 2 - Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Cell 3 - get the code from your PRIVATE GitHub repo (paste a token with read access when asked):
```python
from getpass import getpass
token = getpass("GitHub token: ")
!git clone https://{token}@github.com/YOUR-USERNAME/YOUR-REPO.git /content/repo
%cd /content/repo
!git remote set-url origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
```
Cell 4 - install and copy data:
```python
!pip install -q -r colab/requirements-colab.txt
!mkdir -p /content/data && cp -r /content/drive/MyDrive/logagent/data/processed /content/data/
!ls -la /content/data/processed
```

## B. Pilot (do this first: pipeline test, throughput, resume test)
```python
!python colab/train.py --data-dir /content/data/processed/pilot --output-dir /content/drive/MyDrive/logagent/pilot_run --max-steps 60 --save-steps 20
```
Resume test: run it again, and STOP it (Runtime > Interrupt execution) after "checkpoint-20" exists
(`!ls /content/drive/MyDrive/logagent/pilot_run/checkpoints`). Run the same command once more.
It must print "Resuming from .../checkpoint-20" and carry on from step 20.

## C. Real training (size it from the pilot's measured samples/s)
```python
!python colab/train.py --data-dir /content/data/processed --output-dir /content/drive/MyDrive/logagent/run1 --max-per-class 3000
```
If the session disconnects: reconnect, redo cells 2-4, run the same command again; it resumes.
If you get an out-of-memory error add `--batch-size 2 --grad-accum 8`.

## D. Evaluation (same examples for every model)
```python
P = "/content/drive/MyDrive/logagent/predictions"
A = "/content/drive/MyDrive/logagent/run1/adapter"
D = "/content/data/processed"
!python colab/evaluate_slm.py --data {D}/val_eval.jsonl  --adapter {A} --out {P}/finetuned_val_eval.jsonl
!python colab/evaluate_slm.py --data {D}/test_eval.jsonl --adapter {A} --out {P}/finetuned_test_eval.jsonl
!python colab/evaluate_slm.py --data {D}/test_eval.jsonl --out {P}/untuned_test_eval.jsonl
```
Optional supplementary run on the full test set (slow): `--data {D}/test.jsonl --out {P}/finetuned_test_full.jsonl`.
Download the files in `predictions` to `D:\log-anomaly-agent\results\predictions\`.

## E. Merge for the laptop
```python
!python colab/merge_adapter.py --adapter /content/drive/MyDrive/logagent/run1/adapter --out /content/drive/MyDrive/logagent/merged
```
Download the `merged` folder (about 3 GB) to `D:\log-anomaly-agent\models\merged\`.

## F. Optional: tuning on validation (the report mentions learning rate and LoRA rank)
Short runs, then evaluate each adapter on `val_eval.jsonl`; pick by validation F1 only:
```python
!python colab/train.py --data-dir /content/data/processed --output-dir /content/drive/MyDrive/logagent/tune_r8  --max-per-class 1000 --epochs 1 --rank 8  --alpha 16
!python colab/train.py --data-dir /content/data/processed --output-dir /content/drive/MyDrive/logagent/tune_lr1 --max-per-class 1000 --epochs 1 --lr 1e-4
```