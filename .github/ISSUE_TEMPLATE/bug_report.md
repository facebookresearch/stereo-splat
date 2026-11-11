---
name: Bug report
about: Report a crash, wrong result, or reproducibility problem
title: "[Bug] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of the problem.

## To reproduce

Exact command(s) and config used:

```bash
# e.g. python scripts/evaluate.py --config configs/eval_tartanv2.yaml ...
```

Full error output / traceback:

```
paste here
```

## Expected behavior

What you expected to happen (for metric mismatches: the numbers you got and the
numbers you expected, and which dataset split).

## Environment

- OS:
- Python version:
- PyTorch version and CUDA version (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`):
- gsplat version:
- GPU model:
- StereoSplat commit (`git rev-parse HEAD`):
- FoundationStereo commit and checkpoint used:

## Additional context

Anything else (dataset preparation steps, modified configs, ...).
