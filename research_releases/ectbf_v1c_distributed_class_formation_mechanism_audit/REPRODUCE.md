# Reproduction

## Environment

- Python 3.12.14
- NumPy 2.5.3

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Verify archived files

```bash
python -m unittest -v test_distributed_class_formation_mechanism_audit_v1.py
python audit_distributed_class_formation_mechanism_audit_v1.py \
  results/distributed_class_formation_mechanism_audit_v1_holdout.json
(cd results && sha256sum -c distributed_class_formation_mechanism_audit_v1_holdout.sha256)
sha256sum -c SHA256SUMS
```

Expected formal result SHA-256: `4dc41edd1eaaa91fb352319d4a1158ebe49b3cd9a292c2bff3f78baec5540ba3`

The frozen banks, thresholds, source members, and environment are recorded in `results/distributed_class_formation_mechanism_audit_v1_frozen_manifest.json`.
