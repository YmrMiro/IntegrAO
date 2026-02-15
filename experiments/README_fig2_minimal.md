# Fig.2 Minimal Reproduction

This script targets a minimal, runnable reproduction of IntegrAO trend curves:

- Data: `InterSim` export (preferred) or synthetic 3-view substitute
- Missing setups:
  - `one_complete`: one modality complete, two modalities subsampled by overlap ratio
  - `all_incomplete`: common overlap block + modality-unique split
- Metric: NMI vs overlap

## 1) Environment

Python 3.9/3.10.

Install package dependencies:

```bash
pip install -r requirement.txt
```

Install PyTorch first (CPU or CUDA, according to your machine):

```bash
pip install torch torchvision torchaudio
```

Install graph + explainability dependencies used by IntegrAO:

```bash
pip install torch-geometric captum
```

## 2) Run with synthetic data (quick start)

```bash
python experiments/fig2_minimal_repro.py \
  --data-source synthetic \
  --n-samples 300 \
  --n-clusters 5 \
  --overlaps 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --repeats 10 \
  --epochs 200 \
  --output-dir outputs/fig2_minimal_synth
```

## 3) Run with InterSim CSV exports

Prepare a folder containing:

- `view1.csv`
- `view2.csv`
- `view3.csv`
- `labels.csv` (first column index, second column label)

Then run:

```bash
python experiments/fig2_minimal_repro.py \
  --data-source intersim_csv \
  --intersim-dir path/to/intersim_export \
  --overlaps 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --repeats 10 \
  --epochs 200 \
  --output-dir outputs/fig2_minimal_intersim
```

## 4) Outputs

The script writes:

- `nmi_raw.csv`
- `nmi_summary.csv`
- `nmi_curve.png`

inside `--output-dir`.
