# lcm-explo

[![Release](https://img.shields.io/github/v/release/YHALLOUARD/lcm-explo)](https://img.shields.io/github/v/release/YHALLOUARD/lcm-explo)
[![Build status](https://img.shields.io/github/actions/workflow/status/YHALLOUARD/lcm-explo/main.yml?branch=main)](https://github.com/YHALLOUARD/lcm-explo/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/YHALLOUARD/lcm-explo/branch/main/graph/badge.svg)](https://codecov.io/gh/YHALLOUARD/lcm-explo)
[![Commit activity](https://img.shields.io/github/commit-activity/m/YHALLOUARD/lcm-explo)](https://img.shields.io/github/commit-activity/m/YHALLOUARD/lcm-explo)
[![License](https://img.shields.io/github/license/YHALLOUARD/lcm-explo)](https://img.shields.io/github/license/YHALLOUARD/lcm-explo)

This is an exploration of LCM and SONAR

- **Github repository**: <https://github.com/YHALLOUARD/large_concept_model/>
- **Documentation** <https://YHALLOUARD.github.io/large_concept_model/>

## Getting started

### 1. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

## Tips

Run processing

```bash
uv run python scripts/process_wikipedia.py --output-dir notebooks/data --device cuda --num-articles 10000
```

Run Base lcm training

```bash
uv run python scripts/train_base_lcm.py --embeddings-dir notebooks/data --output-dir notebooks/data/checkpoints --max-epochs 100
```
