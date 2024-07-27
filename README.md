# Usage

```bash
TORCH_CUDA_ARCH_LIST=9.0a+PTX python setup.py develop && python test.py
```

Optionally, do `denoise-h100.sh python test.py` for less noisy (but slower)
results.
